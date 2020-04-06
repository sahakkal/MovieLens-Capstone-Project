# Step 1: Set up the MovieLens10M data into “edx” and “validation” sets in order to test the final recommendation system model with the lowest RMSE. 

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens10M data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove no longer needed objects.
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Step 2: Split the edx data into train and test sets in order to explore different training models.

library(caret)
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Same process used earlier with edx and validation sets:
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(test_index, temp, removed)

# Step 3: Build a function that computes the RMSE for vectors of ratings and their corresponding predictors.

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
  }

# Step 4: Explore 4 models of recommendation systems.

# Just the Average Model:

# Compute the estimate of μ:
mu_hat <- mean(train_set$rating)
mu_hat

# Compute the RMSE:
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# Start a table to keep track of models and RMSEs:
rmse_results <- tibble(Method = "Just the Average Model", RMSE = naive_rmse)
rmse_results
# A tibble: 1 x 2



# Average Plus Movie Effect Model:

# Compute the estimate of b_i: (“hat” notation for estimates dropped from here on)
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Plot histogram of b_i:
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

# Compute the RMSE:
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
MovieEffect_rmse <- RMSE(predicted_ratings, test_set$rating)
MovieEffect_rmse

# Add to earlier table of models and RMSEs:
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Average Plus Movie Effect Model",
                                 RMSE = MovieEffect_rmse))

rmse_results %>% knitr::kable()

  
  
# Average Plus Movie and User Effects Model:
  
# Plot histogram of user effect:
  train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Compute the estimate of b_u:
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Compute the RMSE:
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
M_U_rmse <- RMSE(predicted_ratings, test_set$rating)
M_U_rmse

# Add to earlier table of models and RMSEs:
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Average Plus Movie and User Effects Model", RMSE = M_U_rmse))

rmse_results %>% knitr::kable()

  

# Regularized Average Plus Movie and User Effects Model:
  
# Cross-validation to determine the best lambda for Regularized Average + Movie and User Effects Model:
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + l)) 
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Plot the RMSEs vs. lambdas to see which lambda minimizes the RMSE:
qplot(lambdas, rmses)

# Verify the optimal lambda from the plot:
lambda <- lambdas[which.min(rmses)]
lambda

# Pull predictions based on the Regularized Average + Movie and User Effects Model using the optimal value of lambda:
lambda <- 5

Reg_rmses <- sapply(lambda, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

Reg_rmses

# Add to earlier table of models and RMSEs:
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Regularized Average Plus Movie and User Effects Model", RMSE = Reg_rmses))

rmse_results %>% knitr::kable()
  
# Regularization improved the previous models to an RMSE of 0.8641362.
  
  
# Step 5: Test the final model (Regularized Average + Movie and User Effects) with the edx and the validation sets. 
  
lambda <- 5

Reg_rmses_edx <- sapply(lambda, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred) 
  return(RMSE(predicted_ratings, validation$rating))
})

Reg_rmses_edx

# The recommendation system built using the Regularized Average Plus Movie and User Effects Model made predictions on the validation set with an RMSE of 0.8648177. 
