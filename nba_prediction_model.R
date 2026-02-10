# =============================================================================
# NBA GAME OUTCOME PREDICTION MODEL — 2025-26 SEASON
# =============================================================================
# Predicts: Total Points (O/U), Point Differential (Spread/Winner)
# Method:   XGBoost with Bayesian hyperparameter optimization
# CV:       Time-series aware (rolling origin) cross-validation
# =============================================================================

# ---- 0. SETUP & PACKAGES ----------------------------------------------------

required_pkgs <- c(

  # Data

  "hoopR", "jsonlite", "httr2",
  # Wrangling

  "tidyverse", "lubridate", "zoo", "janitor", "slider", "glue",
  # Modeling
  "tidymodels", "xgboost", "finetune", "probably", "vip",
  # Evaluation & Viz
  "yardstick", "ggplot2", "patchwork", "scales", "knitr"
)

install_if_missing <- function(pkgs) {
  new_pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
  if (length(new_pkgs)) install.packages(new_pkgs, repos = "https://cran.r-project.org")
}
install_if_missing(required_pkgs)
invisible(lapply(required_pkgs, library, character.only = TRUE))

set.seed(2026)
options(scipen = 999)

cat("
====================================================
  NBA Prediction Model — Initialization Complete
====================================================
\n")

# ---- 1. DATA INGESTION ------------------------------------------------------
# Pull team-level box scores for every completed game of the 2025-26 season.
# We try multiple methods in order of reliability.

cat("[1/6] Pulling NBA game data...\n")

game_logs <- tibble()   # will hold our final clean data

# ---- METHOD 0: hoopR::load_nba_team_box (GitHub-cached data) ----
# This pulls pre-cached data from GitHub releases — works from ANY IP
# including cloud/CI environments (GitHub Actions, etc.) where the
# NBA Stats API blocks requests. This is the most reliable method.

cat("   Trying Method 0: load_nba_team_box (GitHub cached)...\n")

game_logs <- tryCatch({
  raw <- hoopR::load_nba_team_box(seasons = 2026)
  if (is.data.frame(raw) && nrow(raw) > 0) {
    df <- raw |> as_tibble() |> clean_names()
    # Standardize column names to match what the rest of the script expects
    # load_nba_team_box uses ESPN-style names; map to NBA Stats style
    name_map <- c(
      "game_id"            = "game_id",
      "game_date"          = "game_date",
      "team_short_display_name" = "team_abbreviation",
      "team_abbreviation"  = "team_abbreviation",
      "team_slug"          = "team_abbreviation",
      "field_goals_made"   = "fgm",
      "field_goals_attempted" = "fga",
      "three_point_field_goals_made" = "fg3m",
      "three_point_field_goals_attempted" = "fg3a",
      "free_throws_made"   = "ftm",
      "free_throws_attempted" = "fta",
      "offensive_rebounds"  = "oreb",
      "defensive_rebounds"  = "dreb",
      "total_rebounds"      = "reb",
      "assists"             = "ast",
      "steals"              = "stl",
      "blocks"              = "blk",
      "turnovers"           = "tov",
      "team_turnovers"      = "tov",
      "points"              = "pts",
      "team_score"          = "pts"
    )
    # Apply renaming for columns that exist
    existing <- intersect(names(name_map), names(df))
    if (length(existing) > 0) {
      for (old_name in existing) {
        new_name <- name_map[old_name]
        if (new_name != old_name && !new_name %in% names(df)) {
          names(df)[names(df) == old_name] <- new_name
        }
      }
    }
    # Ensure game_date is Date type
    if ("game_date" %in% names(df)) {
      df <- df |> mutate(game_date = as.Date(game_date))
    }
    # Normalize ESPN team abbreviations to NBA Stats standard
    abbr_fixes <- c("GS" = "GSW", "SA" = "SAS", "PHO" = "PHX", "NY" = "NYK",
                    "NO" = "NOP", "UTAH" = "UTA", "WSH" = "WAS", "BKN" = "BKN")
    if ("team_abbreviation" %in% names(df)) {
      for (from in names(abbr_fixes)) {
        df$team_abbreviation[df$team_abbreviation == from] <- abbr_fixes[from]
      }
    }
    cat(glue::glue("   ✓ Method 0 succeeded: {nrow(df)} rows.\n"))
    df
  } else {
    tibble()
  }
}, error = function(e) {
  message("   ⚠ Method 0 failed: ", e$message)
  tibble()
})

# ---- METHOD 1: hoopR::nba_leaguegamelog (NBA Stats API wrapper) ----
# This is the cleanest single-call approach. We try several parameter combos
# because hoopR versions differ in argument names.

if (nrow(game_logs) == 0) {
cat("   Trying Method 1: nba_leaguegamelog...\n")

# hoopR uses the ENDING year for season (2025-26 season = 2026)
season_attempts <- list(
  list(season = "2025-26", season_type_all_star = "Regular Season", player_or_team = "T"),
  list(season = 2026,      season_type_all_star = "Regular Season", player_or_team = "T"),
  list(season = "2025-26", season_type = "Regular Season", player_or_team = "T"),
  list(season = 2026,      season_type = "Regular Season", player_or_team = "T")
)

for (i in seq_along(season_attempts)) {
  params <- season_attempts[[i]]
  game_logs_raw <- tryCatch({
    do.call(nba_leaguegamelog, params)
  }, error = function(e) NULL)

  if (!is.null(game_logs_raw)) {
    # Extract the data — hoopR returns either a list with $LeagueGameLog or a dataframe
    if (is.list(game_logs_raw) && !is.data.frame(game_logs_raw)) {
      # It's a named list; find the data element
      for (nm in c("LeagueGameLog", "league_game_log", names(game_logs_raw)[1])) {
        if (nm %in% names(game_logs_raw) && is.data.frame(game_logs_raw[[nm]])) {
          game_logs <- game_logs_raw[[nm]] |> as_tibble() |> clean_names()
          break
        }
      }
      # If still empty, try first element
      if (nrow(game_logs) == 0 && length(game_logs_raw) > 0 && is.data.frame(game_logs_raw[[1]])) {
        game_logs <- game_logs_raw[[1]] |> as_tibble() |> clean_names()
      }
    } else if (is.data.frame(game_logs_raw)) {
      game_logs <- game_logs_raw |> as_tibble() |> clean_names()
    }
  }
  if (nrow(game_logs) > 0) {
    cat(glue::glue("   ✓ Method 1 succeeded (attempt {i}): {nrow(game_logs)} rows.\n"))
    break
  }
}

}

# ---- METHOD 2: Direct NBA Stats API call via httr2 ----
# If hoopR wrapper failed, hit the API directly.

if (nrow(game_logs) == 0) {
  cat("   Method 1 failed. Trying Method 2: Direct NBA Stats API...\n")

  game_logs <- tryCatch({
    resp <- httr2::request("https://stats.nba.com/stats/leaguegamelog") |>
      httr2::req_url_query(
        Season       = "2025-26",
        SeasonType   = "Regular Season",
        PlayerOrTeam = "T",
        LeagueID     = "00",
        Direction    = "ASC",
        Sorter       = "DATE"
      ) |>
      httr2::req_headers(
        `User-Agent`      = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        `Accept`          = "application/json",
        `Accept-Language`  = "en-US,en;q=0.9",
        `Referer`         = "https://www.nba.com/",
        `x-nba-stats-origin` = "stats",
        `x-nba-stats-token`  = "true"
      ) |>
      httr2::req_retry(max_tries = 3, backoff = ~ 2) |>
      httr2::req_perform()

    json <- httr2::resp_body_json(resp)

    # NBA Stats API returns: resultSets -> [0] -> headers + rowSet
    headers <- unlist(json$resultSets[[1]]$headers)
    rows    <- json$resultSets[[1]]$rowSet

    df <- do.call(rbind, lapply(rows, function(r) {
      as.data.frame(matrix(unlist(r), nrow = 1, byrow = TRUE), stringsAsFactors = FALSE)
    }))
    names(df) <- headers
    df |> as_tibble() |> clean_names()
  }, error = function(e) {
    message("   ⚠ Method 2 failed: ", e$message)
    tibble()
  })

  if (nrow(game_logs) > 0) cat(glue::glue("   ✓ Method 2 succeeded: {nrow(game_logs)} rows.\n"))
}

# ---- METHOD 3: hoopR ESPN team box scores (game-by-game) ----
# Slower but more reliable than the NBA Stats API.

if (nrow(game_logs) == 0) {
  cat("   Method 2 failed. Trying Method 3: ESPN team box scores...\n")

  # Use espn_nba_team_box to pull completed games
  game_logs <- tryCatch({
    # Pull the schedule to get game IDs first
    schedule <- espn_nba_scoreboard(season = 2026)
    if (is.data.frame(schedule) && nrow(schedule) > 0) {
      schedule |> as_tibble() |> clean_names()
    } else {
      tibble()
    }
  }, error = function(e) {
    message("   ⚠ Method 3 (schedule) failed: ", e$message)
    tibble()
  })

  if (nrow(game_logs) > 0) cat(glue::glue("   ✓ Method 3 succeeded: {nrow(game_logs)} rows.\n"))
}

# ---- METHOD 4: Load from CSV (manual fallback) ----

if (nrow(game_logs) == 0 && file.exists("nba_game_logs_2025_26.csv")) {
  cat("   Trying Method 4: Loading from local CSV...\n")
  game_logs <- read_csv("nba_game_logs_2025_26.csv", show_col_types = FALSE) |> clean_names()
  if (nrow(game_logs) > 0) cat(glue::glue("   ✓ Loaded {nrow(game_logs)} rows from CSV.\n"))
}

# ---- DIAGNOSTICS: Print what we got ----

if (nrow(game_logs) == 0) {
  stop(paste0(
    "\n\n",
    "====================================================================\n",
    "  ALL DATA PULL METHODS FAILED\n",
    "====================================================================\n",
    "  This usually means the NBA Stats API is blocking requests or\n",
    "  hoopR needs an update.\n\n",
    "  TROUBLESHOOTING:\n",
    "  1. Run: install.packages('hoopR') to get the latest version\n",
    "  2. Check your internet connection\n",
    "  3. Try running in a fresh R session\n",
    "  4. The NBA Stats API sometimes blocks non-browser requests.\n",
    "     Wait a few minutes and retry.\n",
    "  5. Manual fallback: download game logs as CSV from\n",
    "     Basketball Reference (basketball-reference.com) and save as\n",
    "     'nba_game_logs_2025_26.csv' in your working directory.\n",
    "====================================================================\n"
  ))
}

cat(glue::glue("\n   → {nrow(game_logs)} team-game box score rows loaded.\n"))
cat("   Column names: ", paste(names(game_logs)[1:min(10, ncol(game_logs))], collapse = ", "), "...\n")
cat("   Checking a few rows:\n")
print(head(game_logs, 3))
cat("\n")

# ---- 2. DATA CLEANING & STRUCTURING -----------------------------------------

cat("[2/6] Cleaning and structuring data...\n")

# We need a game-level dataframe where each row = one game, with columns for
# both home and away team stats.

# --- 2a. Normalize column names ---
# Different API methods return slightly different column names.
# We standardize to a common set here.

cat("   Detected columns: ", paste(names(game_logs), collapse = ", "), "\n\n")

# Standardize common name variations
name_map <- c(
  "team_abbreviation" = "team_abbreviation",
  "team_name"         = "team_name",
  "game_id"           = "game_id",
  "game_date"         = "game_date",
  "matchup"           = "matchup",
  "wl"                = "wl",
  "w"                 = "wl",                    # some versions use w/l
  "pts"               = "pts",
  "fgm"               = "fgm",
  "fga"               = "fga",
  "fg_pct"            = "fg_pct",
  "fg3m"              = "fg3m",
  "fg3a"              = "fg3a",
  "fg3_pct"           = "fg3_pct",
  "ftm"               = "ftm",
  "fta"               = "fta",
  "ft_pct"            = "ft_pct",
  "oreb"              = "oreb",
  "dreb"              = "dreb",
  "reb"               = "reb",
  "ast"               = "ast",
  "stl"               = "stl",
  "blk"               = "blk",
  "tov"               = "tov",
  "pf"                = "pf",
  "plus_minus"        = "plus_minus"
)

# Check which expected columns exist
expected_stats <- c("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "plus_minus")
missing_cols <- setdiff(expected_stats, names(game_logs))
if (length(missing_cols) > 0) {
  cat("   ⚠ Missing columns (will be set to 0): ", paste(missing_cols, collapse = ", "), "\n")
  for (mc in missing_cols) game_logs[[mc]] <- 0
}

# --- 2b. Parse and type-convert ---

# Detect date format (skip if already Date type)
if (inherits(game_logs$game_date, "Date")) {
  date_fmt <- "already_date"
} else {
  sample_date <- as.character(game_logs$game_date[1])
  if (grepl("^\\d{4}-\\d{2}-\\d{2}", sample_date)) {
    date_fmt <- "%Y-%m-%d"
  } else if (grepl("[A-Z][a-z]{2} \\d", sample_date)) {
    date_fmt <- "%b %d, %Y"
  } else {
    date_fmt <- NULL  # let lubridate guess
  }
}

# Detect home/away - ESPN data uses home_away column, NBA Stats uses matchup
if ("home_away" %in% names(game_logs)) {
  game_logs <- game_logs |> mutate(is_home = (home_away == "home"))
} else if ("matchup" %in% names(game_logs)) {
  game_logs <- game_logs |> mutate(is_home = str_detect(matchup, "vs\\."))
} else {
  stop("Cannot determine home/away status — no 'matchup' or 'home_away' column found.")
}

# Handle date parsing - might already be Date type from Method 0
if (inherits(game_logs$game_date, "Date")) {
  date_parsed <- game_logs$game_date
} else if (!is.null(date_fmt)) {
  date_parsed <- as.Date(game_logs$game_date, format = date_fmt)
} else {
  date_parsed <- lubridate::ymd(game_logs$game_date)
}

games <- game_logs |>
  mutate(
    game_date = date_parsed,
    across(any_of(expected_stats), ~ as.numeric(as.character(.)))
  ) |>
  filter(!is.na(game_date), !is.na(pts), !is.na(is_home))

cat(glue::glue("   → {nrow(games)} valid game rows after cleaning.\n"))
cat(glue::glue("   → Date range: {min(games$game_date)} to {max(games$game_date)}\n"))
cat(glue::glue("   → Teams: {n_distinct(games$team_abbreviation)}\n\n"))

# --- 2c. Separate home and away, then join on game_id ---

home_games <- games |> filter(is_home) |>
  rename_with(~ paste0("home_", .), -c(game_id, game_date))

away_games <- games |> filter(!is_home) |>
  rename_with(~ paste0("away_", .), -c(game_id, game_date))

game_matchups <- inner_join(home_games, away_games, by = c("game_id", "game_date")) |>
  mutate(
    total_points = home_pts + away_pts,
    point_diff   = home_pts - away_pts,         # positive = home win
    home_win     = as.integer(point_diff > 0)
  ) |>
  arrange(game_date)

cat(glue::glue("   → {nrow(game_matchups)} structured game matchups.\n\n"))

# ---- 3. FEATURE ENGINEERING --------------------------------------------------

cat("[3/6] Engineering features...\n")

# We build team-level rolling features, then merge home + away features per game.
# Key principle: features must only use information AVAILABLE BEFORE the game
# (no data leakage).

# --- 3a. Team-level per-game stats to build rolling features from ---

# Ensure wl column exists (ESPN data uses team_winner instead)
if (!"wl" %in% names(games)) {
  if ("team_winner" %in% names(games)) {
    games <- games |> mutate(wl = ifelse(team_winner == TRUE, "W", "L"))
  } else {
    # Fallback: infer from points later; placeholder for now
    games <- games |> mutate(wl = "U")
  }
}

team_game_stats <- games |>
  arrange(team_abbreviation, game_date) |>
  group_by(team_abbreviation) |>
  mutate(
    game_number = row_number(),

    # Possessions estimate (Dean Oliver formula)
    poss = fga - oreb + tov + 0.44 * fta,

    # Four Factors (offense)
    efg_pct     = (fgm + 0.5 * fg3m) / pmax(fga, 1),
    tov_pct     = tov / pmax(poss, 1),
    oreb_pct    = oreb / pmax(oreb + dreb, 1),  # simplified
    ft_rate     = ftm / pmax(fga, 1),

    # Offensive & defensive rating proxies (per 100 possessions)
    off_rating  = pts / pmax(poss, 1) * 100,

    # Rest days
    rest_days   = as.numeric(game_date - lag(game_date, default = game_date[1])),
    rest_days   = pmin(rest_days, 7),            # cap at 7 (long breaks = All-Star etc.)
    is_b2b      = as.integer(rest_days <= 1),

    # Cumulative win pct
    win         = as.integer(wl == "W"),
    cum_wins    = cumsum(lag(win, default = 0)),
    cum_games   = lag(game_number, default = 0),
    win_pct     = cum_wins / pmax(cum_games, 1),

    # Home/away indicator
    is_home     = as.integer(is_home)
  ) |>
  ungroup()

# --- 3b. Rolling averages (lag to prevent leakage) ---
# We compute rolling means over the PRIOR k games (not including current).

rolling_windows <- c(5, 10, 20)

rolling_features_vars <- c(

  "pts", "poss", "efg_pct", "tov_pct", "oreb_pct", "ft_rate",
  "off_rating", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta",
  "oreb", "dreb", "reb", "ast", "stl", "blk", "tov"
)

compute_rolling <- function(df, var, windows) {
  for (w in windows) {
    col_name <- paste0("roll_", w, "_", var)
    df <- df |>
      mutate(
        !!col_name := slider::slide_dbl(
          lag(.data[[var]], 1),    # lag by 1 to exclude current game
          mean, na.rm = TRUE,
          .before = w - 1, .after = 0, .complete = FALSE
        )
      )
  }
  df
}

cat("   Computing rolling averages (5/10/20 game windows)...\n")

team_features <- team_game_stats |>
  group_by(team_abbreviation) |>
  arrange(game_date, .by_group = TRUE)

for (v in rolling_features_vars) {
  team_features <- team_features |>
    group_by(team_abbreviation) |>
    compute_rolling(v, rolling_windows) |>
    ungroup()
}

# --- 3c. Rolling standard deviations (consistency measure, 10-game only) ---
consistency_vars <- c("pts", "off_rating", "efg_pct")

for (v in consistency_vars) {
  col_name <- paste0("roll_sd_10_", v)
  team_features <- team_features |>
    group_by(team_abbreviation) |>
    mutate(
      !!col_name := slider::slide_dbl(
        lag(.data[[v]], 1),
        sd, na.rm = TRUE,
        .before = 9, .after = 0, .complete = FALSE
      )
    ) |>
    ungroup()
}

# --- 3d. Exponentially weighted moving averages (recency-weighted) ---
# Gives more weight to recent games — captures momentum / form.

ema_alpha <- 0.15   # decay factor; higher = more weight on recent

for (v in c("pts", "off_rating", "efg_pct", "tov_pct")) {
  col_name <- paste0("ema_", v)
  team_features <- team_features |>
    group_by(team_abbreviation) |>
    arrange(game_date, .by_group = TRUE) |>
    mutate(
      !!col_name := {
        vals <- lag(.data[[v]], 1)
        ema  <- numeric(n())
        ema[1] <- NA_real_
        for (i in 2:n()) {
          if (is.na(vals[i])) {
            ema[i] <- ema[i - 1]
          } else if (is.na(ema[i - 1])) {
            ema[i] <- vals[i]
          } else {
            ema[i] <- ema_alpha * vals[i] + (1 - ema_alpha) * ema[i - 1]
          }
        }
        ema
      }
    ) |>
    ungroup()
}

cat("   Rolling and EMA features computed.\n")

# --- 3e. Opponent defensive strength (rolling opponent allowed stats) ---
# We need opponent's defensive profile to adjust our offensive expectations.

# Instead of relying on regex-parsed opponent names, we pair teams by game_id.
# Every game_id has exactly 2 rows (one per team). The opponent is the OTHER team.

opp_lookup <- team_game_stats |>
  select(game_id, team_abbreviation) |>
  group_by(game_id) |>
  filter(n() == 2) |>                    # only games with exactly 2 team rows
  mutate(opponent = rev(team_abbreviation)) |>
  ungroup() |>
  select(game_id, team_abbreviation, opponent)

opp_stats <- team_game_stats |>
  select(game_id, team_abbreviation, pts, poss, efg_pct, tov_pct, fg3m, fg3a) |>
  rename_with(~ paste0("opp_", .), -c(game_id, team_abbreviation)) |>
  rename(opponent_team = team_abbreviation)

team_features <- team_features |>
  left_join(opp_lookup, by = c("game_id", "team_abbreviation")) |>
  left_join(opp_stats, by = c("game_id", "opponent" = "opponent_team"))

# Rolling opponent points allowed (defensive strength proxy)
team_features <- team_features |>
  group_by(team_abbreviation) |>
  arrange(game_date, .by_group = TRUE) |>
  mutate(
    roll_10_opp_pts      = slider::slide_dbl(lag(opp_pts, 1), mean, na.rm = TRUE,
                                              .before = 9, .after = 0),
    roll_10_opp_efg      = slider::slide_dbl(lag(opp_efg_pct, 1), mean, na.rm = TRUE,
                                              .before = 9, .after = 0),
    # Defensive rating proxy: opponent points per 100 possessions allowed
    opp_def_rating       = opp_pts / pmax(opp_poss, 1) * 100,
    roll_10_def_rating   = slider::slide_dbl(lag(opp_def_rating, 1), mean, na.rm = TRUE,
                                              .before = 9, .after = 0)
  ) |>
  ungroup()

cat("   Opponent-adjusted features computed.\n")

# --- 3f. Assemble game-level feature matrix ---
# Merge home team features + away team features for each game.

feature_cols <- team_features |>
  select(
    game_id, game_date, team_abbreviation, is_home,

    # Situational
    rest_days, is_b2b, win_pct, game_number,

    # Rolling offensive metrics (5, 10, 20)
    starts_with("roll_5_"), starts_with("roll_10_"), starts_with("roll_20_"),

    # Consistency
    starts_with("roll_sd_"),

    # Momentum (EMA)
    starts_with("ema_"),

    # Opponent-adjusted defensive context
    roll_10_opp_pts, roll_10_opp_efg, roll_10_def_rating
  )

home_feat <- feature_cols |>
  filter(is_home == 1) |>
  rename_with(~ paste0("h_", .), -c(game_id, game_date)) |>
  select(-h_is_home)

away_feat <- feature_cols |>
  filter(is_home == 0) |>
  rename_with(~ paste0("a_", .), -c(game_id, game_date)) |>
  select(-a_is_home)

model_data <- inner_join(home_feat, away_feat, by = c("game_id", "game_date")) |>
  inner_join(
    game_matchups |> select(game_id, game_date, total_points, point_diff, home_win,
                            home_pts, away_pts),
    by = c("game_id", "game_date")
  ) |>
  arrange(game_date)

# --- 3g. Derived interaction features ---
# Matchup-specific: how does team A's offense interact with team B's defense?

model_data <- model_data |>
  mutate(
    # Offensive rating gap
    off_rating_gap_10  = h_roll_10_off_rating - a_roll_10_off_rating,
    # EFG gap
    efg_gap_10         = h_roll_10_efg_pct - a_roll_10_efg_pct,
    # Pace prediction: average of both teams' pace
    pace_proxy_10      = (h_roll_10_poss + a_roll_10_poss) / 2,
    # Combined pace (faster pace → higher total)
    combined_pace_5    = h_roll_5_poss + a_roll_5_poss,
    # Rest advantage
    rest_advantage     = h_rest_days - a_rest_days,
    # Win pct differential
    win_pct_diff       = h_win_pct - a_win_pct,
    # Turnover differential
    tov_diff_10        = a_roll_10_tov_pct - h_roll_10_tov_pct,  # positive = home forces more
    # 3-point volume
    three_volume_10    = h_roll_10_fg3a + a_roll_10_fg3a,
    # Defensive mismatch: home offense vs away defense
    h_off_vs_a_def     = h_ema_off_rating - a_roll_10_def_rating,
    a_off_vs_h_def     = a_ema_off_rating - h_roll_10_def_rating,
    # Combined offensive + defensive mismatch score
    matchup_diff       = h_off_vs_a_def - a_off_vs_h_def
  )

# Drop rows with too many NAs (early-season games without enough rolling history)
min_games_required <- 10
model_data <- model_data |>
  filter(h_game_number > min_games_required & a_game_number > min_games_required)

# Count features
feature_names <- model_data |>
  select(-game_id, -game_date, -total_points, -point_diff, -home_win,
         -home_pts, -away_pts, -h_team_abbreviation, -a_team_abbreviation) |>
  names()

cat(glue::glue("   → {nrow(model_data)} games with {length(feature_names)} features ready for modeling.\n\n"))

# ---- 4. MODELING — XGBOOST WITH BAYESIAN TUNING -----------------------------

cat("[4/6] Training XGBoost models...\n")

# We build TWO models:
#   Model 1: Predict TOTAL POINTS   (for Over/Under)
#   Model 2: Predict POINT DIFF     (for Spread & Winner)

# --- 4a. Prepare modeling dataframe ---

model_df <- model_data |>
  select(game_id, game_date,
         h_team_abbreviation, a_team_abbreviation,
         all_of(feature_names),
         total_points, point_diff, home_win,
         home_pts, away_pts) |>
  drop_na(any_of(feature_names))

cat(glue::glue("   {nrow(model_df)} games after dropping incomplete rows.\n"))

# --- 4b. Time-aware train/test split ---
# Use last ~15% of games as holdout test set (chronological)

split_idx   <- floor(nrow(model_df) * 0.85)
train_data  <- model_df |> slice(1:split_idx)
test_data   <- model_df |> slice((split_idx + 1):n())

cat(glue::glue("   Train: {nrow(train_data)} games | Test: {nrow(test_data)} games\n"))
cat(glue::glue("   Train period: {min(train_data$game_date)} to {max(train_data$game_date)}\n"))
cat(glue::glue("   Test  period: {min(test_data$game_date)} to {max(test_data$game_date)}\n\n"))

# --- 4c. Recipe (preprocessing) ---

base_recipe <- recipe(~ ., data = train_data |> select(all_of(feature_names))) |>
  step_impute_median(all_predictors()) |>  # fill remaining NAs with median

  step_nzv(all_predictors()) |>            # remove near-zero variance

  step_normalize(all_predictors())         # center & scale for regularization

# --- 4d. XGBoost model specification with tunable hyperparameters ---

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  learn_rate     = tune()
) |>
  set_engine("xgboost",
             objective   = "reg:squarederror",
             nthread     = parallel::detectCores() - 1,
             eval_metric = "rmse") |>
  set_mode("regression")

# --- 4e. Rolling-origin cross-validation (time-series aware) ---
# This ensures we never train on future data.

cv_folds <- rolling_origin(

  train_data,
  initial    = floor(nrow(train_data) * 0.6),
  assess     = floor(nrow(train_data) * 0.1),
  skip       = floor(nrow(train_data) * 0.05),
  cumulative = TRUE
)

cat(glue::glue("   {nrow(cv_folds)} rolling-origin CV folds created.\n"))

# --- 4f. Bayesian optimization for hyperparameters ---

xgb_params <- parameters(
  trees(range = c(200, 1500)),
  tree_depth(range = c(3, 10)),
  min_n(range = c(5, 40)),
  loss_reduction(range = c(-5, 1)),      # log10 scale
  sample_prop(range = c(0.5, 0.95)),
  mtry(range = c(10, min(80, length(feature_names)))),
  learn_rate(range = c(-3, -1.2))        # log10 scale: 0.001 to ~0.06
)

# ............................................................................
# MODEL 1: TOTAL POINTS (Over/Under)
# ............................................................................

cat("\n   ── Model 1: Total Points (O/U) ──\n")

total_wf <- workflow() |>
  add_recipe(
    recipe(total_points ~ ., data = train_data |> select(all_of(feature_names), total_points)) |>
      step_impute_median(all_predictors()) |>
      step_nzv(all_predictors())
  ) |>
  add_model(xgb_spec)

cat("   Running Bayesian hyperparameter optimization (this may take a while)...\n")

total_tuned <- tune_bayes(

  total_wf,
  resamples  = cv_folds,
  param_info = xgb_params,
  initial    = 15,           # initial random search points
  iter       = 30,           # Bayesian optimization iterations
  metrics    = metric_set(rmse, mae, rsq),
  control    = control_bayes(
    verbose     = TRUE,
    no_improve  = 10,        # early stopping if no improvement in 10 iters
    save_pred   = FALSE
  )
)

best_total_params <- select_best(total_tuned, metric = "rmse")
cat("   Best Total Points hyperparameters:\n")
print(best_total_params)

total_final_wf <- total_wf |> finalize_workflow(best_total_params)
total_fit      <- total_final_wf |> fit(data = train_data |> select(all_of(feature_names), total_points))

# ............................................................................
# MODEL 2: POINT DIFFERENTIAL (Spread & Winner)
# ............................................................................

cat("\n   ── Model 2: Point Differential (Spread/Winner) ──\n")

diff_wf <- workflow() |>
  add_recipe(
    recipe(point_diff ~ ., data = train_data |> select(all_of(feature_names), point_diff)) |>
      step_impute_median(all_predictors()) |>
      step_nzv(all_predictors())
  ) |>
  add_model(xgb_spec)

cat("   Running Bayesian hyperparameter optimization...\n")

diff_tuned <- tune_bayes(

  diff_wf,
  resamples  = cv_folds,
  param_info = xgb_params,
  initial    = 15,
  iter       = 30,
  metrics    = metric_set(rmse, mae, rsq),
  control    = control_bayes(
    verbose    = TRUE,
    no_improve = 10,
    save_pred  = FALSE
  )
)

best_diff_params <- select_best(diff_tuned, metric = "rmse")
cat("   Best Point Diff hyperparameters:\n")
print(best_diff_params)

diff_final_wf <- diff_wf |> finalize_workflow(best_diff_params)
diff_fit      <- diff_final_wf |> fit(data = train_data |> select(all_of(feature_names), point_diff))

cat("\n   Both models trained.\n\n")

# ---- 5. EVALUATION ON HOLDOUT TEST SET ---------------------------------------

cat("[5/6] Evaluating on holdout test set...\n\n")

# --- 5a. Generate predictions ---

test_features <- test_data |> select(all_of(feature_names))

test_results <- test_data |>
  select(game_id, game_date, h_team_abbreviation, a_team_abbreviation,
         total_points, point_diff, home_win, home_pts, away_pts) |>
  mutate(
    pred_total  = predict(total_fit, new_data = test_features)$.pred,
    pred_diff   = predict(diff_fit,  new_data = test_features)$.pred,
    pred_winner = as.integer(pred_diff > 0)   # 1 = home, 0 = away
  )

# --- 5b. Regression metrics ---

cat("   ┌──────────────────────────────────────────────┐\n")
cat("   │         TOTAL POINTS MODEL (O/U)             │\n")
cat("   ├──────────────────────────────────────────────┤\n")
total_metrics <- test_results |>
  metrics(truth = total_points, estimate = pred_total) |>
  mutate(.estimate = round(.estimate, 3))
print(total_metrics)

cat("\n   ┌──────────────────────────────────────────────┐\n")
cat("   │       POINT DIFFERENTIAL MODEL (Spread)      │\n")
cat("   ├──────────────────────────────────────────────┤\n")
diff_metrics <- test_results |>
  metrics(truth = point_diff, estimate = pred_diff) |>
  mutate(.estimate = round(.estimate, 3))
print(diff_metrics)

# --- 5c. Classification accuracy (Win/Loss) ---

winner_accuracy <- test_results |>
  mutate(correct = pred_winner == home_win) |>
  summarise(
    n_games       = n(),
    correct       = sum(correct),
    accuracy      = mean(correct),
    home_win_rate = mean(home_win)
  )

cat("\n   ┌──────────────────────────────────────────────┐\n")
cat("   │          WINNER PREDICTION ACCURACY           │\n")
cat("   ├──────────────────────────────────────────────┤\n")
cat(glue::glue("   │ Games: {winner_accuracy$n_games}  |  Correct: {winner_accuracy$correct}  |  Accuracy: {round(winner_accuracy$accuracy * 100, 1)}%\n"))
cat(glue::glue("   │ Baseline (always pick home): {round(winner_accuracy$home_win_rate * 100, 1)}%\n"))
cat("   └──────────────────────────────────────────────┘\n\n")

# --- 5d. Simulated O/U performance (if we had lines) ---
# Without actual betting lines, we simulate: if pred_total > actual median total,
# check how often direction is correct. This approximates O/U edge.

median_total <- median(model_df$total_points, na.rm = TRUE)

ou_sim <- test_results |>
  mutate(
    # Simulated line = season median (rough proxy)
    sim_line    = median_total,
    pred_over   = pred_total > sim_line,
    actual_over = total_points > sim_line,
    ou_correct  = pred_over == actual_over
  )

cat(glue::glue("   O/U direction accuracy (vs. season median {round(median_total, 1)}): {round(mean(ou_sim$ou_correct) * 100, 1)}%\n\n"))

# --- 5e. Feature importance ---

cat("   Top 15 features — Total Points model:\n")
total_imp <- total_fit |>
  extract_fit_parsnip() |>
  vip::vi() |>
  slice_max(Importance, n = 15)
print(total_imp)

cat("\n   Top 15 features — Point Diff model:\n")
diff_imp <- diff_fit |>
  extract_fit_parsnip() |>
  vip::vi() |>
  slice_max(Importance, n = 15)
print(diff_imp)

# ---- 6. PREDICTION FUNCTION FOR NEW GAMES -----------------------------------

cat("\n[6/6] Setting up prediction pipeline for upcoming games...\n\n")

#' Predict upcoming games
#'
#' @param home_team 3-letter abbreviation (e.g., "BOS")
#' @param away_team 3-letter abbreviation (e.g., "LAL")
#' @param team_features_df The team_features dataframe
#' @param total_model Fitted total points model
#' @param diff_model Fitted point differential model
#' @return Tibble with predictions

predict_game <- function(home_team, away_team,

                         team_features_df = team_features,
                                          total_model     = total_fit,
                         diff_model      = diff_fit) {

  # Get latest available features for each team
  get_latest <- function(team_abbr) {
    team_features_df |>
      filter(team_abbreviation == team_abbr) |>
      arrange(desc(game_date)) |>
      slice(1) |>
      select(
        team_abbreviation, game_date, rest_days, is_b2b, win_pct, game_number,
        starts_with("roll_5_"), starts_with("roll_10_"), starts_with("roll_20_"),
        starts_with("roll_sd_"), starts_with("ema_"),
        roll_10_opp_pts, roll_10_opp_efg, roll_10_def_rating
      )
  }

  h <- get_latest(home_team) |> rename_with(~ paste0("h_", .), -team_abbreviation) |>
    select(-team_abbreviation)
  a <- get_latest(away_team) |> rename_with(~ paste0("a_", .), -team_abbreviation) |>
    select(-team_abbreviation)

  # Interaction features
  new_game <- bind_cols(h, a) |>
    mutate(
      off_rating_gap_10 = h_roll_10_off_rating - a_roll_10_off_rating,
      efg_gap_10        = h_roll_10_efg_pct - a_roll_10_efg_pct,
      pace_proxy_10     = (h_roll_10_poss + a_roll_10_poss) / 2,
      combined_pace_5   = h_roll_5_poss + a_roll_5_poss,
      rest_advantage    = h_rest_days - a_rest_days,
      win_pct_diff      = h_win_pct - a_win_pct,
      tov_diff_10       = a_roll_10_tov_pct - h_roll_10_tov_pct,
      three_volume_10   = h_roll_10_fg3a + a_roll_10_fg3a,
      h_off_vs_a_def    = h_ema_off_rating - a_roll_10_def_rating,
      a_off_vs_h_def    = a_ema_off_rating - h_roll_10_def_rating,
      matchup_diff      = h_off_vs_a_def - a_off_vs_h_def
    )

  # Predict
  pred_total <- predict(total_model, new_data = new_game)$.pred
  pred_diff  <- predict(diff_model, new_data = new_game)$.pred

  # Estimate individual team scores
  pred_home_pts <- (pred_total + pred_diff) / 2
  pred_away_pts <- (pred_total - pred_diff) / 2

  tibble(
    home_team     = home_team,
    away_team     = away_team,
    pred_total    = round(pred_total, 1),
    pred_diff     = round(pred_diff, 1),
    pred_home_pts = round(pred_home_pts, 1),
    pred_away_pts = round(pred_away_pts, 1),
    pred_winner   = ifelse(pred_diff > 0, home_team, away_team),
    confidence    = round(abs(pred_diff), 1)
  )
}

#' Predict all games on a given date
#' (Requires manually specifying matchups or pulling from schedule API)

predict_slate <- function(matchups_df, ...) {
  # matchups_df should have columns: home_team, away_team
  matchups_df |>
    pmap_dfr(\(home_team, away_team, ...) {
      tryCatch(
        predict_game(home_team, away_team, ...),
        error = function(e) {
          tibble(home_team = home_team, away_team = away_team,
                 pred_total = NA, pred_diff = NA, pred_home_pts = NA,
                 pred_away_pts = NA, pred_winner = NA, confidence = NA)
        }
      )
    })
}

# ---- 7. VISUALIZATION -------------------------------------------------------

cat("Generating diagnostic plots...\n\n")

# --- 7a. Predicted vs Actual: Total Points ---
p1 <- test_results |>
  ggplot(aes(x = total_points, y = pred_total)) +
  geom_point(alpha = 0.5, color = "#1f77b4") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "lm", se = TRUE, color = "#ff7f0e", alpha = 0.3) +
  labs(
    title    = "Total Points: Predicted vs Actual",
    subtitle = glue::glue("RMSE: {round(total_metrics$.estimate[1], 1)} | R²: {round(total_metrics$.estimate[3], 3)}"),
    x = "Actual Total Points", y = "Predicted Total Points"
  ) +
  theme_minimal(base_size = 13)

# --- 7b. Predicted vs Actual: Point Differential ---
p2 <- test_results |>
  ggplot(aes(x = point_diff, y = pred_diff)) +
  geom_point(alpha = 0.5, color = "#2ca02c") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "lm", se = TRUE, color = "#ff7f0e", alpha = 0.3) +
  labs(
    title    = "Point Differential: Predicted vs Actual",
    subtitle = glue::glue("RMSE: {round(diff_metrics$.estimate[1], 1)} | R²: {round(diff_metrics$.estimate[3], 3)}"),
    x = "Actual Point Diff", y = "Predicted Point Diff"
  ) +
  theme_minimal(base_size = 13)

# --- 7c. Residual distribution ---
p3 <- test_results |>
  mutate(resid_total = total_points - pred_total) |>
  ggplot(aes(x = resid_total)) +
  geom_histogram(bins = 30, fill = "#1f77b4", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Total Points Residuals", x = "Actual - Predicted", y = "Count") +
  theme_minimal(base_size = 13)

p4 <- test_results |>
  mutate(resid_diff = point_diff - pred_diff) |>
  ggplot(aes(x = resid_diff)) +
  geom_histogram(bins = 30, fill = "#2ca02c", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Point Diff Residuals", x = "Actual - Predicted", y = "Count") +
  theme_minimal(base_size = 13)

# --- 7d. Feature importance plots ---
p5 <- total_imp |>
  mutate(Variable = fct_reorder(Variable, Importance)) |>
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#1f77b4", alpha = 0.8) +
  labs(title = "Top Features: Total Points Model") +
  theme_minimal(base_size = 12)

p6 <- diff_imp |>
  mutate(Variable = fct_reorder(Variable, Importance)) |>
  ggplot(aes(x = Importance, y = Variable)) +
  geom_col(fill = "#2ca02c", alpha = 0.8) +
  labs(title = "Top Features: Point Diff Model") +
  theme_minimal(base_size = 12)

# Combine and save
combined_plot <- (p1 | p2) / (p3 | p4) / (p5 | p6) +
  plot_annotation(
    title    = "NBA Prediction Model — Diagnostic Dashboard",
    subtitle = glue::glue("Test set: {min(test_data$game_date)} to {max(test_data$game_date)}"),
    theme    = theme(
      plot.title    = element_text(size = 18, face = "bold"),
      plot.subtitle = element_text(size = 12, color = "grey40")
    )
  )

ggsave("nba_model_diagnostics.png", combined_plot, width = 16, height = 18, dpi = 150)
cat("   Saved: nba_model_diagnostics.png\n")

# ---- 8. SAVE MODELS & DATA --------------------------------------------------

save(
  total_fit, diff_fit, team_features, model_data, feature_names,
  total_tuned, diff_tuned, best_total_params, best_diff_params,
  file = "nba_models.RData"
)
cat("   Saved: nba_models.RData\n")

# ---- 9. EXAMPLE USAGE -------------------------------------------------------

cat("
====================================================
  ✓ MODEL READY
====================================================

EXAMPLE — Predict a single game:

  predict_game('BOS', 'LAL')

EXAMPLE — Predict a full slate:

  todays_games <- tibble(
    home_team = c('BOS', 'MIL', 'DEN'),
    away_team = c('LAL', 'NYK', 'PHX')
  )
  predict_slate(todays_games)

EXAMPLE — Load saved models in a new session:

  load('nba_models.RData')

NOTES:
  • The model needs ~10+ games of history per team for reliable rolling features.
  • Re-run this script periodically (weekly) to retrain on latest data.
  • For real O/U and spread evaluation, integrate actual betting lines
    (available via hoopR ESPN data or external APIs like the-odds-api.com).
  • Consider adding injury data for a significant accuracy boost.
  • The model's edge (if any) will be slim — NBA markets are efficient.
    Focus on games where your model disagrees with the line by 3+ points.

")
