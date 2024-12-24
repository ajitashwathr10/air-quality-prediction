library(tidyverse)
library(nonlinearTseries)
library(forecast)
library(keras)
library(tseries)

calculate_lyapunov <- function(time_series, m = 3, d = 1) {
  ts_embedded <- embed(time_series, dimension = m)
  
  distances <- as.matrix(dist(ts_embedded))
  diag(distances) <- Inf
  nearest <- apply(distances, 1, which.min)

  div_rates <- numeric(length(nearest))
  for(i in 1:(length(nearest) - 1)) {
    div_rates[i] <- log(abs(time_series[i + 1] - time_series[nearest[i] + 1]))
  }
  
  mean(div_rates, na.rm = TRUE)
}

generate_chaotic_data <- function(n = 1000) {
  rho <- 28
  sigma <- 10
  beta <- 8/3
  dt <- 0.01
  x <- y <- z <- numeric(n)
  x[1] <- runif(1, -10, 10)
  y[1] <- runif(1, -10, 10)
  z[1] <- runif(1, 0, 30)
  
  for(i in 2:n) {
    dx <- sigma * (y[i-1] - x[i-1])
    dy <- x[i-1] * (rho - z[i-1]) - y[i-1]
    dz <- x[i-1] * y[i-1] - beta * z[i-1]
    
    x[i] <- x[i-1] + dx * dt
    y[i] <- y[i-1] + dy * dt
    z[i] <- z[i-1] + dz * dt
  }
  
  ts_data <- ts(x + rnorm(n, 0, 0.1))
  return(ts_data)
}

build_lstm_model <- function(data, lookback = 10) {
  sequences <- matrix(0, nrow = length(data) - lookback, ncol = lookback)
  for(i in 1:(length(data) - lookback)) {
    sequences[i,] <- data[i:(i + lookback - 1)]
  }
  
  X <- sequences
  y <- data[(lookback + 1):length(data)]
  
  X <- array(X, dim = c(nrow(X), lookback, 1))
  
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(lookback, 1)) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = optimizer_adam(),
    loss = "mse"
  )

  history <- model %>% fit(
    X, y,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 0
  )
  
  return(list(model = model, history = history))
}

main_analysis <- function() {
  set.seed(123)
  data <- generate_chaotic_data(1000)

  lyap_exp <- calculate_lyapunov(data)
  correlation_dim <- corrDim(data, m = 3, d = 1)
 
  decomp <- stl(data, s.window = "periodic")

  model_results <- build_lstm_model(as.numeric(data))
  
  predict_with_intervals <- function(model, X, n_iterations = 100) {
    predictions <- matrix(0, nrow = n_iterations, ncol = length(X))
    
    for(i in 1:n_iterations) {
      predictions[i,] <- predict(model, X)
    }
    
    pred_mean <- colMeans(predictions)
    pred_lower <- apply(predictions, 2, quantile, 0.025)
    pred_upper <- apply(predictions, 2, quantile, 0.975)
    
    return(list(
      mean = pred_mean,
      lower = pred_lower,
      upper = pred_upper
    ))
  }
  
  cat("Chaos Analysis Results:\n")
  cat("Largest Lyapunov Exponent:", lyap_exp, "\n")
  cat("Correlation Dimension:", correlation_dim$dim, "\n")
  
  return(list(
    data = data,
    decomposition = decomp,
    lyapunov = lyap_exp,
    correlation_dim = correlation_dim,
    model = model_results
  ))
}


results <- main_analysis()
ggplot(data.frame(time = 1:length(results$data), value = results$data)) +
  geom_line(aes(x = time, y = value)) +
  labs(title = "Chaotic Air Quality Time Series",
       x = "Time",
       y = "Value") +
  theme_minimal()