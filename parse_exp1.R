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

exp1 <- read_survey("local/Prob+judge+task+1_May+12,+2021_14.24.csv")

df <- exp1 %>% 
  mutate(pass_check = ifelse(choose70=="70%", 1,0)) %>% 
  rename(duration = `Duration (in seconds)`) %>% 
  filter(Status != "Survey Preview", Progress==100) %>% 
  mutate(ID = 1:n()) %>% 
  select(ID, age, gender, race, zip, StartDate, duration, contains("pj"), pass_check) %>% 
  gather("trial_string", "response", contains("pj")) %>% 
  mutate(
    querytype = conds[as.numeric(str_extract(trial_string, "\\d+"))],
    weather_cond = case_when(
      grepl("coldrainy", trial_string) ~ "coldrainy",
      grepl("sunnywarm", trial_string) ~ "sunnywarm"
    ),
    location_cond = case_when(
      grepl("london", trial_string) ~ "london",
      grepl("LA", trial_string) ~ "LA"
    ),
    condition = paste(weather_cond, location_cond, sep="_"),
    response = as.numeric(gsub("%","", response))/100.
    ) %>% 
  select(-trial_string)

write_csv(df, "data/exp1.csv")
