# ============================================
# GRAPHICS IN R
# ============================================
# 
# AGENDA:
# - High-level graphics
# - Custom graphics
# - Layered graphics in ggplot2
# ============================================

# ============================================
# FUNCTIONS FOR GRAPHICS
# ============================================
# The functions hist(), boxplot(), plot(), points(), lines(), text(), mtext(), axis(), etc. 
# form a suite that plot graphs and add features to the graph
#
# Each of these functions have various options, to learn more about them, use the help
# Example: ?hist, ?plot, ?par
#
# par() can be used to set or query graphical parameters
# Example: par(mfrow=c(2,2)) sets up a 2x2 grid of plots
# ============================================

# ============================================
# HIGH-LEVEL GRAPHICS
# ============================================

# ============================================
# UNIVARIATE DATA: HISTOGRAM
# ============================================
# Extract income data from state.x77 dataset
x = state.x77[, 2]  # 50 average state incomes in 1977

# Create plots directory if it doesn't exist
if (!dir.exists("plots")) {
  dir.create("plots")
}

# Save histogram to file
png("plots/state_income_histogram.png", width = 800, height = 600)
hist(x, 
     breaks = 8,
     xlab = "Income",
     main = "Histogram of State Income in 1977",
     col = "lightblue",
     border = "black")
dev.off()

cat("Histogram saved to plots/state_income_histogram.png\n")

# ============================================
# UNIVARIATE DATA: HISTOGRAM (EARTHQUAKE DEPTHS)
# ============================================
# Extract earthquake depth data from quakes dataset
y = quakes$depth  # 1000 earthquake depths

# Save histogram to file
png("plots/earthquake_depth_histogram.png", width = 800, height = 600)
hist(y, 
     breaks = seq(0, 700, by = 70),
     xlab = "Earthquake Depth",
     main = "Histogram of Earthquake Depths",
     col = "lightcoral",
     border = "black")
dev.off()

cat("Histogram saved to plots/earthquake_depth_histogram.png\n")

# ============================================
# EMPIRICAL CDF
# ============================================
# Function ecdf() provides data for empirical cdf

# Plot empirical CDF for state income data
# Can add vertical lines and remove dots
png("plots/state_income_ecdf.png", width = 800, height = 600)
plot.ecdf(x, 
          verticals = T,
          pch = "",
          xlab = "Income",
          main = "ECDF of State Income in 1977")
dev.off()

cat("Empirical CDF saved to plots/state_income_ecdf.png\n")

# Plot empirical CDF for earthquake depth data
png("plots/earthquake_depth_ecdf.png", width = 800, height = 600)
plot.ecdf(y, 
          verticals = T,
          pch = "",
          xlab = "Earthquake Depth",
          main = "ECDF of Earthquake Depths")
dev.off()

cat("Empirical CDF saved to plots/earthquake_depth_ecdf.png\n")

# ============================================
# QQNORM() AND QQPLOT()
# ============================================
# qqnorm() plots the quantiles of a data set against the quantiles of a Normal distribution
# qqplot() plots the quantiles of a first data set against the quantiles of a second data set

# QQ plot for state income data
png("plots/state_income_qqnorm.png", width = 800, height = 600)
qqnorm(x, main = "Q-Q Plot of State Income (1977)")
qqline(x, col = "red")  # red reference line
dev.off()

cat("QQ plot saved to plots/state_income_qqnorm.png\n")

# QQ plot for earthquake depth data
png("plots/earthquake_depth_qqnorm.png", width = 800, height = 600)
qqnorm(y, main = "Q-Q Plot of Earthquake Depths")  # qq plot for the earthquake depths
qqline(y, col = "red")  # red reference line
dev.off()

cat("QQ plot saved to plots/earthquake_depth_qqnorm.png\n")

# ============================================
# BOX PLOTS
# ============================================
# Box plots show distribution of data by groups

# Box plot of insect counts by spray type
png("plots/insect_spray_boxplot.png", width = 800, height = 600)
boxplot(count ~ spray, 
        data = InsectSprays,
        main = "Insect Counts by Spray Type",
        xlab = "Spray",
        ylab = "Count",
        col = "lightgreen")
dev.off()

cat("Box plot saved to plots/insect_spray_boxplot.png\n")

# ============================================
# SCATTERPLOTS: plot(x, y)
# ============================================
# Scatterplots show the relationship between two continuous variables

# Scatterplot of earthquake locations
png("plots/earthquake_locations_scatterplot.png", width = 800, height = 600)
plot(quakes$long, quakes$lat, 
     xlab = "Latitude",
     ylab = "Longitude",
     main = "Location of Earthquake Epicenters",
     pch = 20,
     col = "blue")
dev.off()

cat("Scatterplot saved to plots/earthquake_locations_scatterplot.png\n")

# Scatterplot with symbols scaled by magnitude
png("plots/earthquake_locations_symbols.png", width = 800, height = 600)
symbols(quakes$long, quakes$lat, 
        circles = 10 ^ quakes$mag,
        xlab = "Latitude",
        ylab = "Longitude",
        main = "Location of Earthquake Epicenters",
        inches = 0.3)
dev.off()

cat("Symbol plot saved to plots/earthquake_locations_symbols.png\n")

# ============================================
# THREE-DIMENSIONAL DATA: pairs(x)
# ============================================
# pairs() creates a scatterplot matrix showing relationships between all variable pairs

# Pairs plot for trees dataset (Girth, Height, Volume)
png("plots/trees_pairs_plot.png", width = 800, height = 800)
pairs(trees,
      main = "Pairs Plot of Tree Measurements")
dev.off()

cat("Pairs plot saved to plots/trees_pairs_plot.png\n")

# ============================================
# THREE DIMENSIONAL PLOTS
# ============================================
# contour() creates contour plots for three-dimensional data

# Contour plot of criminal data
png("plots/crimtab_contour_plot.png", width = 800, height = 600)
contour(crimtab, 
        main = "Contour Plot of Criminal Data",
        xlab = "Height",
        ylab = "Finger Length")
dev.off()

cat("Contour plot saved to plots/crimtab_contour_plot.png\n")

# Image plot of criminal data
png("plots/crimtab_image_plot.png", width = 800, height = 600)
image(crimtab, 
      main = "Image Plot of Criminal Data",
      xlab = "Height",
      ylab = "Finger Length")
dev.off()

cat("Image plot saved to plots/crimtab_image_plot.png\n")

# Perspective plot of criminal data
png("plots/crimtab_persp_plot.png", width = 800, height = 600)
persp(crimtab, 
      theta = 30,
      main = "Perspective Plot of Criminal Data",
      xlab = "Height",
      ylab = "Finger Length",
      zlab = "Frequency")
dev.off()

cat("Perspective plot saved to plots/crimtab_persp_plot.png\n")

# ============================================
# CATEGORICAL DATA: PIE CHARTS
# ============================================
# Pie charts display proportions of categorical data

# Create pie chart of pie sales
pie.sales = c(0.12, 0.30, 0.26, 0.16, 0.04, 0.12)
names(pie.sales) = c("Blueberry", "Cherry", "Apple", "Boston Creme",
                     "Other", "Vanilla Creme")

png("plots/pie_sales_chart.png", width = 800, height = 600)
pie(pie.sales, 
    col = c("blue", "red", "green", "wheat", "orange", "white"),
    main = "Pie Sales Distribution")
dev.off()

cat("Pie chart saved to plots/pie_sales_chart.png\n")

# dotchart() and barplot() also available

# Bar plot of Virginia death rates
png("plots/va_deaths_barplot.png", width = 800, height = 600)
barplot(VADeaths, 
        beside = T,
        legend = T,
        main = "Virginia Death Rates per 1000 in 1940",
        xlab = "Population Group",
        ylab = "Death Rate per 1000",
        col = c("lightblue", "mistyrose", "lightcyan", "lavender", "cornsilk"))
dev.off()

cat("Bar plot saved to plots/va_deaths_barplot.png\n")

# ============================================
# TIME SERIES PLOTS
# ============================================
# Time series plots show data over time

# Time series plot of airline passengers
png("plots/airline_passengers_ts.png", width = 800, height = 600)
ts.plot(AirPassengers, 
        xlab = "Date",
        ylab = "Passengers (in thousands)",
        main = "International Airline Passengers",
        col = "blue",
        lwd = 2)
dev.off()

cat("Time series plot saved to plots/airline_passengers_ts.png\n")

# Time series plot of presidential approval ratings
png("plots/presidential_approval_ts.png", width = 800, height = 600)
ts.plot(presidents, 
        xlab = "Date",
        ylab = "Approval Rating",
        main = "Presidential Approval Ratings",
        col = "darkgreen",
        lwd = 2)
dev.off()

cat("Time series plot saved to plots/presidential_approval_ts.png\n")

# ============================================
# BINOMIAL DISTRIBUTION
# ============================================
# Plot of binomial distribution with n=5 and p=.4

x = 0:5
y = dbinom(x, 5, 2 / 5)

png("plots/binomial_distribution_plot.png", width = 800, height = 600)
plot(x, y, 
     type = "h",
     main = "Binomial Distribution (n=5, p=0.4)",
     xlab = "Value",
     ylab = "Probability",
     lwd = 3,
     col = "darkblue")
dev.off()

cat("Binomial distribution plot saved to plots/binomial_distribution_plot.png\n")

# ============================================
# NORMAL DISTRIBUTION
# ============================================
# Probability density function for the standard Normal distribution from -3 to 3

x = seq(-3, 3, by = 0.01)
y = dnorm(x)

png("plots/normal_distribution_plot.png", width = 800, height = 600)
plot(x, y, 
     type = "l",
     main = "Normal Distribution",
     xlab = "x",
     ylab = "f(x)",
     lwd = 2,
     col = "blue")
dev.off()

cat("Normal distribution plot saved to plots/normal_distribution_plot.png\n")

# ============================================
# TWO EMPIRICAL CDFs: PUROMYCIN DATASET
# ============================================
# Compare treated vs untreated enzyme reaction rates

x = Puromycin$rate[Puromycin$state == "treated"]
y = Puromycin$rate[Puromycin$state == "untreated"]

png("plots/puromycin_ecdf_comparison.png", width = 800, height = 600)
plot.ecdf(x, 
          verticals = TRUE,
          pch = "",
          xlim = c(60, 200),
          main = "Treated versus Untreated",
          xlab = "Reaction Rate",
          ylab = "Cumulative Probability")
lines(ecdf(y), 
      verticals = TRUE,
      pch = "",
      xlim = c(60, 200),
      col = "blue")
legend("bottomright", 
       c("Treated", "Untreated"),
       pch = "",
       col = c("black", "blue"),
       lwd = 1)
dev.off()

cat("Puromycin ECDF comparison saved to plots/puromycin_ecdf_comparison.png\n")

# ============================================
# SAVING A PLOT TO A FILE
# ============================================
# Begin with functions postscript(), pdf(), tiff(), jpeg(), png(), etc.
# ... put all your plotting commands here ...
# Finish with dev.off()

# Example: Save the same comparison plot as PDF
pdf("plots/2cdfs.pdf", width = 6, height = 4)
plot.ecdf(x, 
          verticals = TRUE,
          pch = "",
          xlim = c(60, 200),
          main = "Treated versus Untreated",
          xlab = "Reaction Rate",
          ylab = "Cumulative Probability")
lines(ecdf(y), 
      verticals = TRUE,
      pch = "",
      xlim = c(60, 200),
      col = "blue")
legend("bottomright", 
       c("Treated", "Untreated"),
       pch = "",
       col = c("black", "blue"),
       lwd = 1)
dev.off()

cat("PDF saved to plots/2cdfs.pdf\n")

# ============================================
# MULTIPLE PLOTS ON ONE SET OF AXES
# ============================================
# Plot multiple lines on the same graph

x = seq(0, 2 * pi, length = 100)
sine = sin(x)
cosine = cos(x)

png("plots/sine_cosine_plot.png", width = 800, height = 600)
matplot(x, cbind(sine, cosine), 
        col = c(1, 1),
        type = "l",
        lty = c(1, 2),
        lwd = 2,
        main = "Sine and Cosine Functions",
        xlab = "x",
        ylab = "y")
legend("topright", 
       c("sin(x)", "cos(x)"),
       col = c(1, 1),
       lty = c(1, 2),
       lwd = 2)
dev.off()

cat("Sine and cosine plot saved to plots/sine_cosine_plot.png\n")

# ============================================
# MULTIPLE FRAME PLOTS
# ============================================
# Use par(mfrow) to create a grid of plots

png("plots/precipitation_multiplot.png", width = 800, height = 800)
par(mfrow = c(2, 2))
boxplot(precip, main = "Box Plot", ylab = "Precipitation")
hist(precip, main = "Histogram", xlab = "Precipitation")
plot.ecdf(precip, main = "Empirical CDF")
qqnorm(precip, main = "Q-Q Plot")
par(mfrow = c(1, 1))  # Reset to single plot
dev.off()

cat("Multiple frame plot saved to plots/precipitation_multiplot.png\n")

# ============================================
# PLOT USING STATISTICAL MODEL
# ============================================
# Use formula notation to plot relationships from a dataset

png("plots/puromycin_model_plot.png", width = 800, height = 600)
plot(rate ~ conc, 
     data = Puromycin,
     pch = 15 * (state == "treated") + 1,
     main = "Enzyme Reaction Rate vs Substrate Concentration",
     xlab = "Substrate Concentration",
     ylab = "Reaction Rate")
legend("bottomright", 
       legend = c("Untreated", "Treated"),
       pch = c(1, 16))
dev.off()

cat("Statistical model plot saved to plots/puromycin_model_plot.png\n")

# ============================================
# PLOT USING persp() FOR WIRE MESH
# ============================================
# Create 3D surface plots with persp()

x = seq(-8, 8, length = 100)
y = x
f = function(x, y) sin(sqrt(x ^ 2 + y ^ 2)) / (sqrt(x ^ 2 + y ^ 2))
z = outer(x, y, f)

png("plots/persp_wire_mesh.png", width = 800, height = 800)
persp(x, y, z, 
      xlab = "",
      ylab = "",
      zlab = "",
      axes = F,
      box = F,
      theta = 30,
      phi = 30,
      main = "3D Surface: sin(r)/r")
dev.off()

cat("3D wire mesh plot saved to plots/persp_wire_mesh.png\n")

# ============================================
# CUSTOM GRAPHICS
# ============================================
# Custom plot based on Leemis Chapter 21

# ============================================
# CUSTOM PLOT: LEEMIS CHAPTER 21
# ============================================
# Create custom visualization using low-level plotting functions

png("plots/leemis_custom_plot.png", width = 800, height = 600)

# Set up empty plot with specific margins and limits
plot(c(0, 9), c(-10, 20), type = "n", 
     xlab = "", ylab = "",
     main = "", axes = FALSE)

# Add custom axis labels
mtext("c(0, 9)", side = 1, line = 1)
mtext("margin 1", side = 1, line = 2)
mtext("margin 2", side = 2, line = 2)
mtext("margin 4", side = 4, line = 2)

# Add right/top side labels
axis(4, at = seq(0, 20, length.out = 8), 
     labels = paste0("line ", 0:7), 
     las = 1, tick = FALSE)

# Add text annotations with different fonts
text(2, 18, "bold font", font = 2, pos = 3)
text(2, 16, "italics font", font = 3, pos = 3)
text(2, 14, "bold & italics font", font = 4, pos = 3)
text(2, 12, expression(alpha * beta * Gamma), pos = 3)

# Add annotation for right/top justified
text(7.5, 15, "right/top justified", pos = 3)
text(7.5, 15, "+", cex = 1.5)

# Add annotation for left/bottom justified  
text(2, -9, "left/bottom justified", pos = 1)

# Add curves and lines
curve(x^2 - 5*x + 2, from = 0, to = 8, add = TRUE, lwd = 2)
curve(-0.5*x + 3, from = 0, to = 8, add = TRUE, lty = 2, lwd = 2)

# Add "slanted Text" label
text(2.5, -2, "slanted Text", srt = 15)

# Add polygon (yellow triangle)
polygon(c(0, 1, 0), c(-3, 0, 2), col = "yellow", border = "black")

# Add "plotting region" label with arrow
text(5, 9, "plotting region", pos = 3)
arrows(4.5, 9, 3, 7, length = 0.1)

# Add mathematical expression
text(6, -5, expression(frac(lambda[i], 2^x)), cex = 1.5)

# Add points
points(6, 3, pch = 1, cex = 1.5)
points(4, -9, pch = 1, cex = 1)

# Add "w" in blue
text(7.5, 3, "w", col = "blue", font = 2)

# Add "8" on right side
text(8.5, 3, "8")

dev.off()

cat("Leemis custom plot saved to plots/leemis_custom_plot.png\n")

# ============================================
# GGPLOT2
# ============================================
# Everything so far has been part of base R
# The ggplot2 package is a popular package by Hadley Wickham
# Based on the grammar of graphics, which tries to take the good parts of 
# base and lattice graphics and none of the bad parts
# http://ggplot2.org

# ============================================
# FORCED EXPIRATORY VOLUME (FEV) DATA
# ============================================
# Explore a data on the relationship between smoking and pulmonary function 
# from Rosner (1999) using layered graphics created with ggplot2. 
# The data consists of a sample of 654 youths, aged 3 to 19, in the area of 
# East Boston during middle to late 1970's. Our main interest is in the 
# relationship between smoking and FEV.

# Load the data and ggplot2
load(url("http://www.faculty.ucr.edu/~jflegal/fev.RData"))
library(ggplot2)

# Display structure of the data
cat("\n=== FEV Data Structure ===\n")
str(fevdata)
cat("\n")

# ============================================
# LAYERED GRAPHICS IN GGPLOT2
# ============================================
# ggplot2 allows you to construct multi-layered graphics. 
# A plot in ggplot2 consists of several components:
#   - Defaults
#   - Layers
#   - Scales
#   - Coordinate system
#
# Layers consist of:
#   - Data
#   - Mapping
#   - Geom
#   - Stat
#   - Position

# ggplot2 uses the + operator to build up a plot from these components. 
# The basic plot definition looks like this:
#
# ggplot(data, mapping) + 
#   layer(
#     stat = "",
#     geom = "",
#     position = "", 
#     geom_params = list(), 
#     stat_params = list(),
#   )
#
# We usually won't write out the full specification of layer, but use shortcuts like:
#   - geom_point()
#   - stat_summary()
#
# Every geom has a default stat and every stat has a default geom.

# Usually, data and mappings are the same for all layers and so they can be 
# stored as defaults:
#
# ggplot(data, mapping = aes(x = x, y = y))
#
# All layers use the default values of data and mapping unless you override 
# them explicitly. The aes() function describes the mapping that will be used 
# for each layer. You must specify a default, but you can also specify per 
# layer mappings and data:
#
# ggplot(data, mapping = aes(x = x, y = y)) + 
#   geom_point(aes(color = z)) + 
#   geom_line(data = another_data)

# ============================================
# LAYERING
# ============================================
# You can add additional layers to a plot with the + operator. 
# Let's try adding a line that shows the average value of fev for each age. 
# One way to do this is to construct an additional data frame with columns 
# corresponding to age and average value of fev and then add a layer with 
# this data. We will do this with the dplyr package.

# Install and load dplyr (install only if needed)
if (!require(dplyr, quietly = TRUE)) {
  install.packages("dplyr")
}
library(dplyr)

# Create base plot with default data and mapping
s <- ggplot(fevdata, aes(x = age, y = fev))

# Calculate mean fev for each age group
fev_mean <- summarize(group_by(fevdata, age), fev = mean(fev))

# Create layered plot: points + line showing average
png("plots/fev_layered_plot.png", width = 800, height = 600)
s + geom_point() + geom_line(data = fev_mean)
dev.off()

cat("FEV layered plot saved to plots/fev_layered_plot.png\n")

# ============================================
# SMOOTHING
# ============================================
# Similarly, we can add a smoother to the scatterplot by first computing 
# the smooth and storing it in a data frame. Then add a layer with that data. 
# Since smoothers are so useful, this operation is available in ggplot2 as a stat.
#
# stat_smooth() provides a smoothing transformation. It creates a new data 
# frame with the values of the smooth and by default uses geom="ribbon" so 
# that both the smooth curve and error bands are shown.

# Basic smooth (default is loess)
png("plots/fev_smooth_default.png", width = 800, height = 600)
s + geom_point() + stat_smooth()
dev.off()
cat("FEV smooth (default loess) saved to plots/fev_smooth_default.png\n")

# The default smoother is loess. This is the name given to the locally 
# weighted quadratic regression smoother with tricubic weight function. 
# Its bandwidth can be specified indirectly with the span parameter.

# Smooth with span = 1
png("plots/fev_smooth_span1.png", width = 800, height = 600)
s + geom_point() + stat_smooth(span = 1)
dev.off()
cat("FEV smooth (span=1) saved to plots/fev_smooth_span1.png\n")

# Smooth with span = 1/2
png("plots/fev_smooth_span_half.png", width = 800, height = 600)
s + geom_point() + stat_smooth(span = 1/2)
dev.off()
cat("FEV smooth (span=0.5) saved to plots/fev_smooth_span_half.png\n")

# We could also use linear regression by specifying lm
png("plots/fev_smooth_lm.png", width = 800, height = 600)
s + geom_point() + stat_smooth(method = 'lm')
dev.off()
cat("FEV smooth (linear model) saved to plots/fev_smooth_lm.png\n")

# ============================================
# GROUPING
# ============================================
# Clearly, we can see age and fev are highly correlated. What else is 
# correlated with age and fev?
# How can we compare the relationship between age and fev among smokers 
# and non-smokers? One way is to use two separate smoothers: one for 
# smokers and one for non-smokers.
# Can do this using the group aesthetic. It specifies that we wish to 
# group the data according by some variable before layering.

# Group by smoke
p <- ggplot(fevdata, aes(x = age, y = fev, group = smoke))
png("plots/fev_grouped_smoke.png", width = 800, height = 600)
p + geom_point() + stat_smooth()
dev.off()
cat("FEV grouped by smoke saved to plots/fev_grouped_smoke.png\n")

# Group by smoke with color
png("plots/fev_grouped_smoke_color.png", width = 800, height = 600)
p + geom_point() + stat_smooth(aes(color = smoke))
dev.off()
cat("FEV grouped by smoke (colored smooths) saved to plots/fev_grouped_smoke_color.png\n")

# Color for points and smooths
p <- ggplot(fevdata, aes(x = age, y = fev, group = smoke, color = smoke))
png("plots/fev_grouped_smoke_all_color.png", width = 800, height = 600)
p + geom_point() + stat_smooth()
dev.off()
cat("FEV grouped by smoke (all colored) saved to plots/fev_grouped_smoke_all_color.png\n")

# How is the following plot different from the others in terms of the 
# conclusions you might draw about the relation between smoke and fev?
png("plots/fev_grouped_smoke_lm.png", width = 800, height = 600)
p + geom_point() + stat_smooth(method = 'lm')
dev.off()
cat("FEV grouped by smoke (linear) saved to plots/fev_grouped_smoke_lm.png\n")

# ============================================
# FACETING
# ============================================
# Faceting creates small multiples: separate plots for subsets of data

p <- ggplot(fevdata, aes(x = height, y = fev)) + facet_grid(sex ~ age)
png("plots/fev_faceted_sex_age.png", width = 1200, height = 600)
p + geom_point()
dev.off()
cat("FEV faceted by sex and age saved to plots/fev_faceted_sex_age.png\n")

# One problem with the previous plot is that there are too many ages and 
# relatively few observations at each age. We can instead try dividing age 
# into a smaller number of groups using the cut() function. cut() creates a 
# new factor variable by cutting its input. Here we cut age into 5 intervals 
# of equal length:

fevdata <- transform(fevdata, age_group = cut(age, breaks = 5))

# Then make new plots with age groups
p <- ggplot(fevdata, aes(x = height, y = fev)) + facet_grid(sex ~ age_group)
png("plots/fev_faceted_sex_age_group.png", width = 1200, height = 600)
p + geom_point(aes(color = smoke))
dev.off()
cat("FEV faceted by sex and age_group (colored by smoke) saved to plots/fev_faceted_sex_age_group.png\n")

# ============================================
# SUMMARY
# ============================================
# R has strong graphic capabilities
# Graphing is an iterative process; Don't rely on the default options
# Avoid gimmicks, use the minimum amount of ink to get your point across
# A small table can be better than a large graph
# Carefully consider the size and shape of your graph, bigger is not always better

cat("\n=== Graphics Tutorial Complete ===\n")
cat("All plots saved to plots/ directory\n")
