# =============================================================================
# NBA DAILY PREDICTION SCRIPT
# =============================================================================
# Run this AFTER the main model has been trained and saved (nba_models.RData).
# Automatically pulls today's schedule and generates predictions.
# =============================================================================

library(tidyverse)
library(hoopR)
library(httr2)
library(jsonlite)
library(glue)

# ---- 1. Load saved models and data ------------------------------------------

if (!file.exists("nba_models.RData")) {
  stop("No saved models found. Run nba_prediction_model.R first.")
}
load("nba_models.RData")
cat("✓ Models loaded.\n\n")

# ---- 2. Auto-pull today's schedule ------------------------------------------

today <- Sys.Date()
cat(glue("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"))
cat(glue("  NBA PREDICTIONS — {format(today, '%A, %B %d, %Y')}\n"))
cat(glue("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"))

todays_games <- tibble()

# --- Method 1: ESPN API direct (most reliable) ---
cat("   Pulling today's schedule...\n")

todays_games <- tryCatch({
  date_str <- format(today, "%Y%m%d")
  url <- glue("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}")
  
  resp <- request(url) |>
    req_retry(max_tries = 3, backoff = ~ 2) |>
    req_perform()
  
  json <- resp_body_json(resp)
  events <- json$events
  
  if (length(events) > 0) {
    map_dfr(events, function(ev) {
      competitors <- ev$competitions[[1]]$competitors
      home_idx <- which(map_chr(competitors, ~ .x$homeAway) == "home")
      away_idx <- which(map_chr(competitors, ~ .x$homeAway) == "away")
      tibble(
        home_team = competitors[[home_idx]]$team$abbreviation,
        away_team = competitors[[away_idx]]$team$abbreviation
      )
    })
  } else {
    tibble()
  }
}, error = function(e) {
  cat(glue("   ⚠ ESPN API failed: {e$message}\n"))
  tibble()
})

if (nrow(todays_games) > 0) {
  cat(glue("   ✓ Found {nrow(todays_games)} games via ESPN API.\n\n"))
}

# --- Method 2: ESPN Scoreboard via hoopR ---
if (nrow(todays_games) == 0) {
  cat("   Trying hoopR ESPN scoreboard...\n")
  
  todays_games <- tryCatch({
    sb <- espn_nba_scoreboard(today)
    if (is.data.frame(sb) && nrow(sb) > 0) {
      sb_clean <- sb |> as_tibble() |> janitor::clean_names()
      cols <- names(sb_clean)
      
      home_col <- cols[str_detect(cols, "home") & str_detect(cols, "abb|abbreviation")]
      away_col <- cols[str_detect(cols, "away") & str_detect(cols, "abb|abbreviation")]
      
      if (length(home_col) >= 1 && length(away_col) >= 1) {
        sb_clean |> select(home_team = !!home_col[1], away_team = !!away_col[1])
      } else {
        cat("   ⚠ ESPN returned data but couldn't parse team abbreviations.\n")
        cat("   Columns: ", paste(cols, collapse = ", "), "\n")
        tibble()
      }
    } else {
      tibble()
    }
  }, error = function(e) {
    cat(glue("   ⚠ hoopR scoreboard failed: {e$message}\n"))
    tibble()
  })
  
  if (nrow(todays_games) > 0) {
    cat(glue("   ✓ Found {nrow(todays_games)} games via hoopR.\n\n"))
  }
}

# --- Method 3: NBA.com schedule CDN ---
if (nrow(todays_games) == 0) {
  cat("   Trying NBA.com schedule API...\n")
  
  todays_games <- tryCatch({
    resp <- request("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json") |>
      req_headers(
        `User-Agent` = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        `Referer`    = "https://www.nba.com/"
      ) |>
      req_retry(max_tries = 2) |>
      req_perform()
    
    schedule <- resp_body_json(resp)
    game_dates <- schedule$leagueSchedule$gameDates
    
    today_games_raw <- NULL
    for (gd in game_dates) {
      gd_date <- as.Date(str_extract(gd$gameDate, "\\d{2}/\\d{2}/\\d{4}"), format = "%m/%d/%Y")
      if (!is.na(gd_date) && gd_date == today) {
        today_games_raw <- gd$games
        break
      }
    }
    
    if (!is.null(today_games_raw) && length(today_games_raw) > 0) {
      map_dfr(today_games_raw, function(g) {
        tibble(
          home_team = g$homeTeam$teamTricode,
          away_team = g$awayTeam$teamTricode
        )
      })
    } else {
      tibble()
    }
  }, error = function(e) {
    cat(glue("   ⚠ NBA.com API failed: {e$message}\n"))
    tibble()
  })
  
  if (nrow(todays_games) > 0) {
    cat(glue("   ✓ Found {nrow(todays_games)} games via NBA.com.\n\n"))
  }
}

# --- Fallback: Manual entry ---
if (nrow(todays_games) == 0) {
  cat("
   ⚠ Could not auto-detect today's schedule from any source.
     This may mean there are no NBA games scheduled today,
     or all APIs are temporarily down.
   
   To enter games manually, uncomment the block below and re-run:

   todays_games <- tibble(
     home_team = c('BOS', 'MIL', 'DEN'),
     away_team = c('LAL', 'NYK', 'PHX')
   )
   
   TEAM ABBREVIATIONS:
     ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND
     LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX
     POR SAC SAS TOR UTA WAS

")
}

# ---- 3. Normalize abbreviations ---------------------------------------------
# ESPN and NBA.com sometimes use slightly different abbreviations.

abbr_normalize <- c(
  "GS"  = "GSW", "SA"  = "SAS", "NO"  = "NOP", "NY"  = "NYK",
  "PHO" = "PHX", "UTAH"= "UTA", "WSH" = "WAS"
)

normalize_abbr <- function(abbr) {
  ifelse(abbr %in% names(abbr_normalize), abbr_normalize[abbr], abbr)
}

if (nrow(todays_games) > 0) {
  todays_games <- todays_games |>
    mutate(
      home_team = normalize_abbr(home_team),
      away_team = normalize_abbr(away_team)
    )
}

# ---- 4. Predict function (self-contained) -----------------------------------
# Defined here so the daily script works standalone with just nba_models.RData.

if (!exists("predict_game")) {
  
  predict_game <- function(home_team, away_team,
                           team_features_df = team_features,
                           total_model      = total_fit,
                           diff_model       = diff_fit) {
    
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
    
    if (nrow(h) == 0) stop(glue("No data found for home team: {home_team}"))
    if (nrow(a) == 0) stop(glue("No data found for away team: {away_team}"))
    
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
    
    pred_total <- predict(total_model, new_data = new_game)$.pred
    pred_diff  <- predict(diff_model, new_data = new_game)$.pred
    
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
}

# ---- 5. Generate predictions ------------------------------------------------

if (nrow(todays_games) == 0) {
  cat("No games to predict. Exiting.\n")
} else {
  
  cat(glue("Predicting {nrow(todays_games)} games...\n\n"))
  
  available_teams <- unique(team_features$team_abbreviation)
  
  predictions <- todays_games |>
    pmap_dfr(function(home_team, away_team) {
      if (!(home_team %in% available_teams)) {
        cat(glue("   ⚠ {home_team} not found in model data. Skipping.\n"))
        return(tibble(home_team = home_team, away_team = away_team,
                      pred_total = NA, pred_diff = NA, pred_home_pts = NA,
                      pred_away_pts = NA, pred_winner = NA, confidence = NA))
      }
      if (!(away_team %in% available_teams)) {
        cat(glue("   ⚠ {away_team} not found in model data. Skipping.\n"))
        return(tibble(home_team = home_team, away_team = away_team,
                      pred_total = NA, pred_diff = NA, pred_home_pts = NA,
                      pred_away_pts = NA, pred_winner = NA, confidence = NA))
      }
      
      tryCatch(
        predict_game(home_team, away_team),
        error = function(e) {
          cat(glue("   ⚠ Error predicting {away_team} @ {home_team}: {e$message}\n"))
          tibble(home_team = home_team, away_team = away_team,
                 pred_total = NA, pred_diff = NA, pred_home_pts = NA,
                 pred_away_pts = NA, pred_winner = NA, confidence = NA)
        }
      )
    })
  
  # ---- 6. Display results ----------------------------------------------------
  
  valid_preds <- predictions |> filter(!is.na(pred_total))
  
  if (nrow(valid_preds) == 0) {
    cat("\n⚠ No valid predictions generated. Check team abbreviations.\n")
    cat("   Teams in model: ", paste(sort(available_teams), collapse = ", "), "\n")
  } else {
    
    cat("\n┌──────────────────────────────────────────────────────────────────────┐\n")
    cat("│                        GAME PREDICTIONS                             │\n")
    cat("├──────────────────────────────────────────────────────────────────────┤\n\n")
    
    valid_preds |>
      mutate(
        matchup = glue("{away_team} @ {home_team}"),
        score   = glue("{pred_away_pts} - {pred_home_pts}"),
        spread  = ifelse(pred_diff > 0,
                         glue("{home_team} -{abs(pred_diff)}"),
                         glue("{away_team} -{abs(pred_diff)}")),
        ou_line = glue("O/U {pred_total}")
      ) |>
      select(matchup, pred_winner, score, spread, ou_line, confidence) |>
      print(n = 30, width = Inf)
    
    # ---- Betting insights ----
    
    cat("\n┌──────────────────────────────────────────────────────────────────────┐\n")
    cat("│                        BETTING INSIGHTS                             │\n")
    cat("├──────────────────────────────────────────────────────────────────────┤\n\n")
    
    strong_winners <- valid_preds |> filter(confidence >= 5)
    if (nrow(strong_winners) > 0) {
      cat("  🏀 HIGH CONFIDENCE WINNERS (spread ≥ 5 pts):\n")
      strong_winners |>
        mutate(
          loser = ifelse(pred_winner == home_team, away_team, home_team),
          pick  = glue("     → {pred_winner} over {loser} (model spread: {confidence})")
        ) |>
        pull(pick) |>
        walk(~ cat(.x, "\n"))
      cat("\n")
    }
    
    tossups <- valid_preds |> filter(confidence < 2.5)
    if (nrow(tossups) > 0) {
      cat("  🎲 TOSS-UPS (spread < 2.5 — look for underdog ML value):\n")
      tossups |>
        mutate(pick = glue("     → {away_team} @ {home_team} (spread: {pred_diff})")) |>
        pull(pick) |>
        walk(~ cat(.x, "\n"))
      cat("\n")
    }
    
    cat("  📊 TOTALS OUTLOOK:\n")
    highest <- valid_preds |> slice_max(pred_total, n = 1, with_ties = FALSE)
    lowest  <- valid_preds |> slice_min(pred_total, n = 1, with_ties = FALSE)
    cat(glue("     Highest total:  {highest$away_team} @ {highest$home_team} → {highest$pred_total}\n"))
    cat(glue("     Lowest total:   {lowest$away_team} @ {lowest$home_team} → {lowest$pred_total}\n"))
    
    cat("\n  💡 STRATEGY:\n")
    cat("     • Compare pred_total vs posted O/U line. Bet when model disagrees by 4+.\n")
    cat("     • Compare pred_diff vs posted spread. Bet when model disagrees by 3+.\n")
    cat("     • High confidence winners → moneyline parlays.\n")
    cat("     • Always cross-reference injury reports before placing bets.\n")
    
    cat("\n└──────────────────────────────────────────────────────────────────────┘\n\n")
  }
  
  # ---- 7. Save to CSV --------------------------------------------------------
  
  output_file <- glue("predictions_{format(today, '%Y_%m_%d')}.csv")
  write_csv(predictions, output_file)
  cat(glue("Predictions saved to: {output_file}\n\n"))
}
