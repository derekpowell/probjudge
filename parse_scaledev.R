## Process raw qualtrics file to a de-identified and formatted csv
## processing the data in R because it's just a bit easier

library(tidyverse)
library(qualtRics)

conds <- c(
  "A",
  "B",
  "notA",
  "notB",
  "AandB",
  "notAandB",
  "AandnotB",
  "notAandnotB",
  "AorB",
  "notAorB",
  "AornotB",
  "notAornotB",
  "AgB",
  "notAgB",
  "AgnotB",
  "notAgnotB",
  "BgA",
  "notBgA",
  "BgnotA",
  "notBgnotA"
)

x <- read_survey("local/Prob+judge+-+scale+testing_June+16,+2021_17.08.csv")

df <- x %>% 
  filter(DistributionChannel=="anonymous") %>% 
  filter(check_item=="Somewhat confident") %>% # filter failing checks
  rename(duration = `Duration (in seconds)`) %>% 
  filter(Status != "Survey Preview", Progress==100) %>% 
  mutate(ID = 1:n()) %>% 
  select(ID, age, gender, condition, race, zip, StartDate, duration, contains("pj"), contains("conf")) %>% # select relevant variables
  gather("trial_string", "response", contains("pj")) %>% # drop systematically missing responses
  filter(!is.na(response)) %>% 
  mutate(trial_string = gsub("pj_?[0-9]","pj",trial_string)) %>%  # unify spread responses
  spread(trial_string, response) %>% 
  pivot_longer(matches("pj|conf"), names_to=c("trial_num", "block", "type", ".value"), names_pattern="([0-9]+)_(b[1-3])_(guided|slider|likert)_(.*)") %>% 
  filter(condition==type) %>% 
  mutate(
    pj = case_when(
      grepl("than", pj) ~ NA_character_,
      grepl("over", pj) ~ NA_character_,
      TRUE ~ gsub("\\%","", pj)
    ),
    pj = as.numeric(pj)/100,
    conf = case_when(
      conf == "Very confident" ~ 3,
      conf == "Confident" ~ 2,
      conf == "Somewhat confident" ~ 1,
      TRUE ~ 0
    )
  )
  # filter(!is.na(response)) %>% 
  # mutate(
  #   # trial_id = , # get trial number
  #   # block_id = , # get block number
  #   # response = , # as.numeric and drop "bad" guided responses (make NA)
  #     )
  # mutate(
  #   querytype = conds[as.numeric(str_extract(trial_string, "\\d+"))],
  #   weather_cond = case_when(
  #     grepl("coldrainy", trial_string) ~ "coldrainy",
  #     grepl("sunnywarm", trial_string) ~ "sunnywarm"
  #   ),
  #   location_cond = case_when(
  #     grepl("london", trial_string) ~ "london",
  #     grepl("LA", trial_string) ~ "LA"
  #   ),
  #   condition = paste(weather_cond, location_cond, sep="_"),
  #   response = as.numeric(gsub("%","", response))/100.
  # ) %>% 
  # select(-trial_string)

write_csv(df, "data/pilot-scaledev.csv")
