library(tidyverse)
library(nonLinearTseries)
library(forecast)
library(keras)
library(tseries)

#Lyapunov exponents
calculate_lyapunov <- function(time_series, m = 3, d = 1) {
    #Embedding dimension (m) and time delay (d)
    ts_embedded <- stats::embed(time_series, dimension = m)
    #Distances between points
    distances <- as.matrix(dist(ts_embedded))
    #Nearest neighbors
    diag(distances) <- Inf
    nearest <- apply(distances, 1, which.min())

    #Divergence rates
    div_rates <- numeric(length(nearest))
    for(i in 1:(length(nearest) - 1)) {
        div_rates[i] <- log(abs(time_series[i + 1] - time_series[nearest[i] + 1]))
    }
    #Estimating largest Lyapunov exponent
    mean(div_rates, na.rm = TRUE)
}

#Generating synthetic air quality data with Chaos Theory
generate_chaotic_data <- function(n = 1000) {
    # System parameters
    rho <- 28
    sigma <- 10
    beta <- 8/3
    dt <- 0.01

    # Arrays
    x <- y <- z <- numeric(n)
    x[1] <- runif(1, -10, 10)
    y[1] <- runif(1, -10, 10)
    z[1] <- runif(1, 0, 30)

    #Chaotic trajectory
    for(i in 2:n) {
        dx <- sigma * (y[i - 1] - x[i - 1])
        dy <- x[i - 1] * (rho - z[i - 1]) - y[i - 1]
        dz <- x[i - 1] * y[i - 1] - beta * z[i - 1]

        x[i] <- x[i - 1] + dx * dt
        y[i] <- y[i - 1] + dy * dt
        z[i] <- z[i - 1] + dz * dt
    }

    ts_data <- ts(x + rnorm(n, 0, 0.1))
    return(ts_data)
}

