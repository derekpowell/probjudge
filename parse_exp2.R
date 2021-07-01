## Process raw qualtrics file to a de-identified and formatted csv
## processing the data in R because it's just a bit easier

library(tidyverse)
library(qualtRics)

get_trial_label <- function(trial_string){
  str_match( attr(exp2[[trial_string]], "label"), "([A-z]+) -")[2]
}


exp2 <- read_survey("local/Prob+judge+task+2_June+26,+2021_10.38.csv")

## NOTE: ID column is extremely important as it is used for indexing of parameters
## inside the model. So this must be 0:n_subj without any missing values for 
## the final data that will be analyzed!

df <- exp2 %>% 
  mutate(pass_check = ifelse(choose70=="70%", 1,0)) %>% 
  rename(duration = `Duration (in seconds)`) %>% 
  filter(Status != "Survey Preview", Progress==100, pass_check==1) %>% 
  # filter(lived_in_london=="No") %>%  # forgot to put this in preregistration
  mutate(ID = 0:(n()-1)) %>% # fix indexing for modeling
  select(ID, age, gender, race, zip, StartDate, duration, contains("pj"), pass_check) %>% 
  gather("trial_string", "response", contains("pj")) %>% 
  mutate(
    querytype = map_chr(trial_string, get_trial_label),
    weather_cond = case_when(
      grepl("coldrainy", trial_string) ~ "coldrainy",
      grepl("sunnywarm", trial_string) ~ "sunnywarm"
    ),
    location_cond = case_when(
      grepl("london", trial_string) ~ "london",
      grepl("yourzip", trial_string) ~ "yourzip"
    ),
    condition = paste(weather_cond, location_cond, sep="_"),
    response = as.numeric(gsub("%","", response))/100.
    ) %>% 
  select(-trial_string)

write_csv(df, "data/exp2-passing.csv")
