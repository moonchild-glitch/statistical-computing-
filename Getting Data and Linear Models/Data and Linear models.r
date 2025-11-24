# R script for "Getting Data and Linear Models"

# ============================================
# GETTING DATA AND LINEAR MODELS
# ============================================
# 
# AGENDA:
# - Getting data into and out of R
# - Using data frames for statistical purposes
# - Introduction to linear models
# ============================================

# Create plots directory if it doesn't exist
if (!dir.exists("../plots")) {
  dir.create("../plots")
}

# ============================================
# READING DATA FROM R
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("READING DATA FROM R\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("You can load and save R objects")
print("R has its own format for this, which is shared across operating systems")
print("It's an open, documented format if you really want to pry into it")
print("")
print("save(thing, file='name') saves thing in a file called name (conventional extension: rda or Rda)")
print("load('name') loads the object or objects stored in the file called name, with their old names")

# Example: Load GMP data, add population column, save and reload
gmp <- read.table("http://faculty.ucr.edu/~jflegal/206/gmp.dat")
gmp$pop <- round(gmp$gmp/gmp$pcgmp)
print("\nGMP data loaded:")
print(head(gmp))

save(gmp, file="gmp.Rda")
print("\nSaved gmp to gmp.Rda")

rm(gmp)
print(paste("exists('gmp') after rm():", exists("gmp")))

not_gmp <- load(file="gmp.Rda")
print(paste("\nColumn names:", paste(colnames(gmp), collapse=", ")))
print(paste("load() returned:", paste(not_gmp, collapse=", ")))

print("\nNote: We can load or save more than one object at once")
print("This is how RStudio will load your whole workspace when you're starting,")
print("and offer to save it when you're done")

# ============================================
# LOADING PACKAGE DATA
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("LOADING PACKAGE DATA\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Many packages come with saved data objects")
print("There's the convenience function data() to load them")

library(MASS)
data(cats, package="MASS")
print("\nSummary of cats data:")
print(summary(cats))

# ============================================
# NON-R DATA TABLES
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("NON-R DATA TABLES\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Tables full of data, just not in the R file format")
print("\nMain function: read.table()")
print("- Presumes space-separated fields, one line per row")
print("- Main argument is the file name or URL")
print("- Returns a dataframe")
print("- Lots of options for things like field separator, column names,")
print("  forcing or guessing column types, skipping lines at the start of the file...")
print("")
print("read.csv() is a short-cut to set the options for reading comma-separated value (CSV) files")
print("Spreadsheets will usually read and write CSV")

# ============================================
# WRITING DATAFRAMES
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("WRITING DATAFRAMES\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Counterpart functions write.table(), write.csv() write a dataframe into a file")
print("\nDrawback: takes a lot more disk space than what you get from load or save")
print("Advantage: can communicate with other programs, or even edit manually")

# Example: Write cats data
write.csv(cats, file="cats_data.csv", row.names=FALSE)
print("\nWrote cats data to cats_data.csv")

# Read it back
cats_from_csv <- read.csv("cats_data.csv")
print("Read cats data back from CSV:")
print(head(cats_from_csv))

# ============================================
# LESS FRIENDLY DATA FORMATS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("LESS FRIENDLY DATA FORMATS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("The foreign package on CRAN has tools for reading data files")
print("from lots of non-R statistical software")
print("")
print("Spreadsheets are special - full of ugly irregularities:")
print("- Values or formulas?")
print("- Headers, footers, side-comments, notes")
print("- Columns change meaning half-way down")

# ============================================
# SPREADSHEETS, IF YOU HAVE TO
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("SPREADSHEETS, IF YOU HAVE TO\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Options for dealing with spreadsheets:")
print("1. Save the spreadsheet as a CSV; read.csv()")
print("2. Save the spreadsheet as a CSV; edit in a text editor; read.csv()")
print("3. Use read.xls() from the gdata package")
print("   - Tries very hard to work like read.csv(), can take a URL or filename")
print("   - Can skip down to the first line that matches some pattern, select different sheets, etc.")
print("   - You may still need to do a lot of tidying up after")

# ============================================
# SO YOU'VE GOT A DATA FRAME
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("SO YOU'VE GOT A DATA FRAME - WHAT CAN WE DO WITH IT?\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("What can we do with it?")
print("- Plot it: examine multiple variables and distributions")
print("- Test it: compare groups of individuals to each other")
print("- Check it: does it conform to what we'd like for our needs")

# Example: Explore the cats data
png("../plots/data_cats_exploration.png", width=1200, height=800)
par(mfrow=c(2,2))

# 1. Body weight distribution
hist(cats$Bwt, main="Distribution of Body Weight", 
     xlab="Body Weight (kg)", col="lightblue", breaks=15)

# 2. Heart weight distribution
hist(cats$Hwt, main="Distribution of Heart Weight",
     xlab="Heart Weight (g)", col="lightgreen", breaks=15)

# 3. Relationship between body and heart weight
plot(cats$Bwt, cats$Hwt, main="Heart Weight vs Body Weight",
     xlab="Body Weight (kg)", ylab="Heart Weight (g)",
     pch=19, col=ifelse(cats$Sex=="F", "red", "blue"))
legend("topleft", legend=c("Female", "Male"), 
       col=c("red", "blue"), pch=19)

# 4. Boxplot by sex
boxplot(Hwt ~ Sex, data=cats, main="Heart Weight by Sex",
        xlab="Sex", ylab="Heart Weight (g)",
        col=c("pink", "lightblue"))

par(mfrow=c(1,1))
dev.off()
print("\nPlot saved: data_cats_exploration.png")

# ============================================
# INTRODUCTION TO LINEAR MODELS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("INTRODUCTION TO LINEAR MODELS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Linear models are one of the most fundamental tools in statistics")
print("In R, we use the lm() function to fit linear models")
print("")
print("Basic syntax: lm(response ~ predictor, data=dataframe)")

# Simple linear regression: Heart weight vs Body weight
model1 <- lm(Hwt ~ Bwt, data=cats)
print("\nModel 1: Heart Weight ~ Body Weight")
print(summary(model1))

png("../plots/data_linear_model1.png", width=800, height=600)
plot(cats$Bwt, cats$Hwt, main="Linear Model: Heart Weight ~ Body Weight",
     xlab="Body Weight (kg)", ylab="Heart Weight (g)",
     pch=19, col="gray")
abline(model1, col="red", lwd=2)
text(2.5, 18, paste("R² =", round(summary(model1)$r.squared, 3)), col="red")
dev.off()
print("Plot saved: data_linear_model1.png")

# Multiple regression: including Sex
model2 <- lm(Hwt ~ Bwt + Sex, data=cats)
print("\nModel 2: Heart Weight ~ Body Weight + Sex")
print(summary(model2))

png("../plots/data_linear_model2.png", width=800, height=600)
plot(cats$Bwt, cats$Hwt, main="Linear Model with Sex: Heart Weight ~ Body Weight + Sex",
     xlab="Body Weight (kg)", ylab="Heart Weight (g)",
     pch=19, col=ifelse(cats$Sex=="F", "red", "blue"))
# Add regression lines for each sex
cats_f <- cats[cats$Sex=="F",]
cats_m <- cats[cats$Sex=="M",]
abline(lm(Hwt ~ Bwt, data=cats_f), col="red", lwd=2)
abline(lm(Hwt ~ Bwt, data=cats_m), col="blue", lwd=2)
legend("topleft", legend=c("Female", "Male"), 
       col=c("red", "blue"), pch=19, lwd=2)
text(2.5, 18, paste("R² =", round(summary(model2)$r.squared, 3)), col="black")
dev.off()
print("Plot saved: data_linear_model2.png")

# ============================================
# MODEL DIAGNOSTICS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("MODEL DIAGNOSTICS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("It's important to check model assumptions:")
print("1. Linearity: Is the relationship actually linear?")
print("2. Homoscedasticity: Is the variance constant?")
print("3. Normality: Are the residuals normally distributed?")
print("4. Independence: Are observations independent?")

png("../plots/data_model_diagnostics.png", width=1200, height=1200)
par(mfrow=c(2,2))
plot(model2)
par(mfrow=c(1,1))
dev.off()
print("\nPlot saved: data_model_diagnostics.png")

# ============================================
# PREDICTIONS FROM LINEAR MODELS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("PREDICTIONS FROM LINEAR MODELS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Once we have a fitted model, we can make predictions")
print("Use the predict() function with new data")

# Create new data for prediction
new_cats <- data.frame(
  Bwt = c(2.0, 2.5, 3.0, 3.5),
  Sex = c("F", "F", "M", "M")
)

predictions <- predict(model2, newdata=new_cats, interval="confidence")
print("\nPredictions for new cats:")
print(cbind(new_cats, predictions))

# ============================================
# COMPARING MODELS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("COMPARING MODELS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("We can compare models using ANOVA")
anova_result <- anova(model1, model2)
print("\nANOVA comparing Model 1 (Bwt only) vs Model 2 (Bwt + Sex):")
print(anova_result)

if(anova_result$`Pr(>F)`[2] < 0.05) {
  print("\nConclusion: Adding Sex significantly improves the model (p < 0.05)")
} else {
  print("\nConclusion: Adding Sex does not significantly improve the model")
}

# ============================================
# SUMMARY
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Key takeaways:")
print("1. R can read/write data in multiple formats (RDA, CSV, etc.)")
print("2. save() and load() for R objects; write.csv() and read.csv() for text")
print("3. Data frames are the primary structure for statistical analysis")
print("4. lm() fits linear models with formula syntax: response ~ predictors")
print("5. Check model diagnostics before trusting results")
print("6. Use predict() to make predictions from fitted models")
print("7. Compare models with anova() or information criteria (AIC, BIC)")

cat("\nGetting Data and Linear Models Tutorial Complete\n")
print(paste("Generated", length(list.files("../plots", pattern="data_.*\\.png")), "plots"))

# Clean up temporary files
if(file.exists("gmp.Rda")) file.remove("gmp.Rda")
if(file.exists("cats_data.csv")) file.remove("cats_data.csv")
print("Cleaned up temporary files")

# ============================================
# TEST CASE: BIRTH WEIGHT DATA
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("TEST CASE: BIRTH WEIGHT DATA\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

library(MASS)
data(birthwt)

print("Original birth weight data summary:")
print(summary(birthwt))

# ============================================
# FROM R HELP
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("FROM R HELP\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Go to R help for more info, because someone documented this data")
print("Try: help(birthwt)")
print("")
print("Original column names:")
print(colnames(birthwt))

# ============================================
# MAKE IT READABLE
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("MAKE IT READABLE\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Rename columns for clarity
colnames(birthwt) <- c("birthwt.below.2500", "mother.age", 
                       "mother.weight", "race",
                       "mother.smokes", "previous.prem.labor", 
                       "hypertension", "uterine.irr",
                       "physician.visits", "birthwt.grams")

print("Renamed columns:")
print(colnames(birthwt))

# Make factors more descriptive
birthwt$race <- factor(c("white", "black", "other")[birthwt$race])
birthwt$mother.smokes <- factor(c("No", "Yes")[birthwt$mother.smokes + 1])
birthwt$uterine.irr <- factor(c("No", "Yes")[birthwt$uterine.irr + 1])
birthwt$hypertension <- factor(c("No", "Yes")[birthwt$hypertension + 1])

print("\nTransformed birth weight data summary:")
print(summary(birthwt))

# ============================================
# EXPLORE IT
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("EXPLORE IT\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Plot 1: Count of Mother's Race
png("../plots/data_birthwt_race.png", width=800, height=600)
plot(birthwt$race, main="Count of Mother's Race in Springfield MA, 1986",
     col=c("lightblue", "lightgreen", "pink"),
     xlab="Race", ylab="Count")
dev.off()
print("Plot saved: data_birthwt_race.png")

# Plot 2: Sorted Mother's Ages
png("../plots/data_birthwt_ages.png", width=800, height=600)
plot(sort(birthwt$mother.age), 
     main="(Sorted) Mother's Ages in Springfield MA, 1986",
     ylab="Mother's Age", xlab="Index",
     pch=19, col="darkblue")
dev.off()
print("Plot saved: data_birthwt_ages.png")

# Plot 3: Birth Weight by Mother's Age
png("../plots/data_birthwt_by_age.png", width=800, height=600)
plot(birthwt$mother.age, birthwt$birthwt.grams,
     main="Birth Weight by Mother's Age in Springfield MA, 1986",
     xlab="Mother's Age", ylab="Birth Weight (g)",
     pch=19, col="darkred")
# Add a smooth trend line
age_order <- order(birthwt$mother.age)
lines(lowess(birthwt$mother.age[age_order], birthwt$birthwt.grams[age_order]), 
      col="blue", lwd=2)
dev.off()
print("Plot saved: data_birthwt_by_age.png")

# ============================================
# EXPLORATORY PLOTS - MULTIPLE VARIABLES
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("EXPLORATORY ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

png("../plots/data_birthwt_exploration.png", width=1200, height=1200)
par(mfrow=c(2,2))

# Birth weight by smoking status
boxplot(birthwt.grams ~ mother.smokes, data=birthwt,
        main="Birth Weight by Smoking Status",
        xlab="Mother Smokes", ylab="Birth Weight (g)",
        col=c("lightgreen", "salmon"))

# Birth weight by race
boxplot(birthwt.grams ~ race, data=birthwt,
        main="Birth Weight by Race",
        xlab="Race", ylab="Birth Weight (g)",
        col=c("lightblue", "lightgreen", "pink"))

# Birth weight by hypertension
boxplot(birthwt.grams ~ hypertension, data=birthwt,
        main="Birth Weight by Hypertension",
        xlab="Hypertension", ylab="Birth Weight (g)",
        col=c("lightblue", "orange"))

# Birth weight by uterine irritability
boxplot(birthwt.grams ~ uterine.irr, data=birthwt,
        main="Birth Weight by Uterine Irritability",
        xlab="Uterine Irritability", ylab="Birth Weight (g)",
        col=c("lightblue", "red"))

par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_exploration.png")

# ============================================
# LINEAR MODEL FOR BIRTH WEIGHT
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("LINEAR MODEL FOR BIRTH WEIGHT\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Simple linear model
bw_model1 <- lm(birthwt.grams ~ mother.age, data=birthwt)
print("Model 1: Birth Weight ~ Mother's Age")
print(summary(bw_model1))

# Multiple regression with key factors
bw_model2 <- lm(birthwt.grams ~ mother.age + mother.weight + mother.smokes + 
                  race + hypertension + uterine.irr, data=birthwt)
print("\nModel 2: Birth Weight ~ Multiple Predictors")
print(summary(bw_model2))

# Compare models
print("\nModel Comparison (ANOVA):")
print(anova(bw_model1, bw_model2))

# ============================================
# MODEL VISUALIZATION
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("MODEL VISUALIZATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

png("../plots/data_birthwt_model.png", width=1200, height=800)
par(mfrow=c(2,3))

# Plot model effects
plot(birthwt$mother.age, birthwt$birthwt.grams,
     main="Birth Weight vs Mother's Age",
     xlab="Mother's Age", ylab="Birth Weight (g)",
     pch=19, col="gray")
abline(bw_model1, col="red", lwd=2)

# Plot residuals vs fitted
plot(bw_model2$fitted.values, bw_model2$residuals,
     main="Residuals vs Fitted",
     xlab="Fitted Values", ylab="Residuals",
     pch=19, col="blue")
abline(h=0, col="red", lwd=2, lty=2)

# Q-Q plot
qqnorm(bw_model2$residuals, main="Normal Q-Q Plot")
qqline(bw_model2$residuals, col="red", lwd=2)

# Scale-Location plot
plot(bw_model2$fitted.values, sqrt(abs(bw_model2$residuals)),
     main="Scale-Location",
     xlab="Fitted Values", ylab="√|Residuals|",
     pch=19, col="purple")

# Residuals vs Mother's Age
plot(birthwt$mother.age, bw_model2$residuals,
     main="Residuals vs Mother's Age",
     xlab="Mother's Age", ylab="Residuals",
     pch=19, col="darkgreen")
abline(h=0, col="red", lwd=2, lty=2)

# Histogram of residuals
hist(bw_model2$residuals, main="Distribution of Residuals",
     xlab="Residuals", col="lightblue", breaks=20)

par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_model.png")

# ============================================
# KEY FINDINGS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("KEY FINDINGS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("From the birth weight analysis:")
print("")

# Extract significant coefficients from model 2
coef_summary <- summary(bw_model2)$coefficients
significant <- coef_summary[coef_summary[,4] < 0.05, ]

print("Significant predictors (p < 0.05):")
if(nrow(significant) > 0) {
  for(i in 1:nrow(significant)) {
    var_name <- rownames(significant)[i]
    coef_val <- significant[i, 1]
    p_val <- significant[i, 4]
    print(sprintf("  %s: coefficient = %.2f, p-value = %.4f", 
                  var_name, coef_val, p_val))
  }
} else {
  print("  No significant predictors at p < 0.05 level")
}

print("")
print(sprintf("Model R-squared: %.4f", summary(bw_model2)$r.squared))
print(sprintf("Adjusted R-squared: %.4f", summary(bw_model2)$adj.r.squared))

cat("\nData and Linear Models Tutorial Complete\n")
total_plots <- length(list.files("../plots", pattern="data_.*\\.png"))
print(paste("Total plots generated:", total_plots))

# ============================================
# BASIC STATISTICAL TESTING
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("BASIC STATISTICAL TESTING\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Let's fit some models to the data pertaining to our outcome(s) of interest")

# Plot: Birth weight by smoking status
png("../plots/data_birthwt_smoking_box.png", width=800, height=600)
plot(birthwt$mother.smokes, birthwt$birthwt.grams, 
     main="Birth Weight by Mother's Smoking Habit", 
     ylab="Birth Weight (g)", xlab="Mother Smokes",
     col=c("lightgreen", "salmon"))
dev.off()
print("Plot saved: data_birthwt_smoking_box.png")

# Two-sample t-test
print("\nTough to tell! Simple two-sample t-test:")
t_test_result <- t.test(birthwt$birthwt.grams[birthwt$mother.smokes == "Yes"], 
                        birthwt$birthwt.grams[birthwt$mother.smokes == "No"])
print(t_test_result)

# ============================================
# LINEAR MODEL COMPARISONS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("LINEAR MODEL COMPARISONS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Does this difference match the linear model?")

# Model 1: Birth weight ~ Smoking
linear.model.1 <- lm(birthwt.grams ~ mother.smokes, data=birthwt)
print("\nLinear Model 1: birthwt.grams ~ mother.smokes")
print(linear.model.1)
print("\nSummary:")
print(summary(linear.model.1))

# Model 2: Birth weight ~ Mother's age
linear.model.2 <- lm(birthwt.grams ~ mother.age, data=birthwt)
print("\nLinear Model 2: birthwt.grams ~ mother.age")
print(linear.model.2)
print("\nSummary:")
print(summary(linear.model.2))

# Diagnostic plots for model 2
print("\nR tries to make diagnostics as easy as possible")
png("../plots/data_birthwt_model2_diagnostics.png", width=1200, height=1200)
par(mfrow=c(2,2))
plot(linear.model.2)
par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_model2_diagnostics.png")

# ============================================
# DETECTING OUTLIERS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("DETECTING OUTLIERS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Note the oldest mother and her heaviest child are greatly skewing this analysis")
print(paste("Maximum mother age:", max(birthwt$mother.age)))
outlier_idx <- which.max(birthwt$mother.age)
print(paste("Birth weight for oldest mother:", birthwt$birthwt.grams[outlier_idx], "grams"))

# Remove outlier (mother.age > 40)
birthwt.noout <- birthwt[birthwt$mother.age <= 40,]
print(paste("\nDataset after removing outliers:", nrow(birthwt.noout), "observations"))

linear.model.3 <- lm(birthwt.grams ~ mother.age, data=birthwt.noout)
print("\nLinear Model 3 (no outliers): birthwt.grams ~ mother.age")
print(linear.model.3)
print("\nSummary:")
print(summary(linear.model.3))

# ============================================
# MORE COMPLEX MODELS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("MORE COMPLEX MODELS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Model 3a: Add smoking behavior
print("Add in smoking behavior:")
linear.model.3a <- lm(birthwt.grams ~ mother.smokes + mother.age, data=birthwt.noout)
print("\nLinear Model 3a: birthwt.grams ~ mother.smokes + mother.age")
print(summary(linear.model.3a))

png("../plots/data_birthwt_model3a_diagnostics.png", width=1200, height=1200)
par(mfrow=c(2,2))
plot(linear.model.3a)
par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_model3a_diagnostics.png")

# Model 3b: Add race with interaction
print("\nAdd in race with interaction:")
linear.model.3b <- lm(birthwt.grams ~ mother.age + mother.smokes*race, data=birthwt.noout)
print("\nLinear Model 3b: birthwt.grams ~ mother.age + mother.smokes*race")
print(summary(linear.model.3b))

png("../plots/data_birthwt_model3b_diagnostics.png", width=1200, height=1200)
par(mfrow=c(2,2))
plot(linear.model.3b)
par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_model3b_diagnostics.png")

# ============================================
# INCLUDING EVERYTHING
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("INCLUDING EVERYTHING\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Let's include everything on this new data set:")
linear.model.4 <- lm(birthwt.grams ~ ., data=birthwt.noout)
print("\nLinear Model 4: birthwt.grams ~ . (all predictors)")
print(linear.model.4)

print("\nWarning: Be careful! One of those variables birthwt.below.2500 is a function of the outcome")

# Model 4a: Exclude the derived variable
linear.model.4a <- lm(birthwt.grams ~ . - birthwt.below.2500, data=birthwt.noout)
print("\nLinear Model 4a: birthwt.grams ~ . - birthwt.below.2500")
print(summary(linear.model.4a))

png("../plots/data_birthwt_model4a_diagnostics.png", width=1200, height=1200)
par(mfrow=c(2,2))
plot(linear.model.4a)
par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_model4a_diagnostics.png")

# ============================================
# GENERALIZED LINEAR MODELS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("GENERALIZED LINEAR MODELS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Maybe a linear increase in birth weight is less important than if it's")
print("below a threshold like 2500 grams (5.5 pounds).")
print("Let's fit a generalized linear model instead:")

# GLM with binomial family (logistic regression)
glm.0 <- glm(birthwt.below.2500 ~ . - birthwt.grams, 
             data=birthwt.noout, family=binomial)
print("\nGLM Model: birthwt.below.2500 ~ . - birthwt.grams")
print(summary(glm.0))

png("../plots/data_birthwt_glm_diagnostics.png", width=1200, height=1200)
par(mfrow=c(2,2))
plot(glm.0)
par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_glm_diagnostics.png")

# ============================================
# MODEL COMPARISON SUMMARY
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("MODEL COMPARISON SUMMARY\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Compare R-squared values
models <- list(
  "Model 1 (smoke)" = linear.model.1,
  "Model 2 (age)" = linear.model.2,
  "Model 3 (age, no outlier)" = linear.model.3,
  "Model 3a (smoke + age)" = linear.model.3a,
  "Model 3b (age + smoke*race)" = linear.model.3b,
  "Model 4a (all vars)" = linear.model.4a
)

print("Linear Model Comparison (R-squared values):")
for(name in names(models)) {
  r2 <- summary(models[[name]])$r.squared
  adj_r2 <- summary(models[[name]])$adj.r.squared
  print(sprintf("  %s: R² = %.4f, Adj R² = %.4f", name, r2, adj_r2))
}

# Best model based on adjusted R-squared
adj_r2_values <- sapply(models, function(m) summary(m)$adj.r.squared)
best_model_name <- names(which.max(adj_r2_values))
print(sprintf("\nBest linear model (by Adj R²): %s", best_model_name))

# GLM metrics
print("\nGLM Model (Logistic Regression):")
print(sprintf("  AIC: %.2f", glm.0$aic))
print(sprintf("  Deviance: %.2f", glm.0$deviance))
print(sprintf("  Null Deviance: %.2f", glm.0$null.deviance))

# Calculate pseudo R-squared for GLM
pseudo_r2 <- 1 - (glm.0$deviance / glm.0$null.deviance)
print(sprintf("  Pseudo R² (McFadden): %.4f", pseudo_r2))

# ============================================
# PREDICTIONS FROM GLM
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("PREDICTIONS FROM GLM\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Make predictions on the data
predictions <- predict(glm.0, type="response")
predicted_class <- ifelse(predictions > 0.5, 1, 0)

# Confusion matrix
actual <- birthwt.noout$birthwt.below.2500
confusion_matrix <- table(Predicted=predicted_class, Actual=actual)
print("Confusion Matrix:")
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(sprintf("\nAccuracy: %.2f%%", accuracy * 100))

# Sensitivity and Specificity
if(sum(actual == 1) > 0) {
  sensitivity <- confusion_matrix[2,2] / sum(actual == 1)
  print(sprintf("Sensitivity (True Positive Rate): %.2f%%", sensitivity * 100))
}
if(sum(actual == 0) > 0) {
  specificity <- confusion_matrix[1,1] / sum(actual == 0)
  print(sprintf("Specificity (True Negative Rate): %.2f%%", specificity * 100))
}

# ============================================
# FINAL SUMMARY
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("FINAL SUMMARY\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Key findings from the birth weight analysis:")
print("")
print("1. T-test shows significant difference in birth weight between")
print("   smoking and non-smoking mothers (p < 0.01)")
print("")
print("2. Mother's age alone is not a significant predictor of birth weight")
print("")
print("3. Outliers (very old mothers) can significantly affect results")
print("")
print("4. Smoking, race, mother's weight, hypertension, and uterine")
print("   irritability are all significant predictors")
print("")
print("5. The full model (Model 4a) explains about 26% of variance")
print("   in birth weight (Adj R² ≈ 0.26)")
print("")
print("6. For predicting low birth weight (<2500g), logistic regression")
print("   (GLM) is more appropriate than linear regression")
print("")
print(sprintf("7. The GLM achieves %.1f%% accuracy in classifying", accuracy * 100))
print("   low birth weight cases")

cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("DATA AND LINEAR MODELS TUTORIAL COMPLETE\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

final_plot_count <- length(list.files("../plots", pattern="data_.*\\.png"))
print(paste("Total plots generated:", final_plot_count))

# ============================================
# PREDICTION OF TEST DATA
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("PREDICTION OF TEST DATA\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Creating train/test split for validation")

# Create train/test split
set.seed(123)
train_indices <- sample(1:nrow(birthwt.noout), size=0.7*nrow(birthwt.noout))
birthwt.train <- birthwt.noout[train_indices, ]
birthwt.test <- birthwt.noout[-train_indices, ]

print(sprintf("Training set: %d observations", nrow(birthwt.train)))
print(sprintf("Test set: %d observations", nrow(birthwt.test)))

# Fit model on training data
train.model <- lm(birthwt.grams ~ . - birthwt.below.2500, data=birthwt.train)
print("\nModel trained on training data:")
print(summary(train.model))

# Predict on test data
birthwt.predict.out <- predict(train.model, newdata=birthwt.test)

print("\nPrediction statistics:")
print(sprintf("Mean predicted value: %.2f grams", mean(birthwt.predict.out)))
print(sprintf("Mean actual value: %.2f grams", mean(birthwt.test$birthwt.grams)))

# Calculate prediction metrics
residuals_test <- birthwt.test$birthwt.grams - birthwt.predict.out
rmse <- sqrt(mean(residuals_test^2))
mae <- mean(abs(residuals_test))
r2_test <- 1 - sum(residuals_test^2) / sum((birthwt.test$birthwt.grams - mean(birthwt.test$birthwt.grams))^2)

print(sprintf("\nTest set performance:"))
print(sprintf("  RMSE: %.2f grams", rmse))
print(sprintf("  MAE: %.2f grams", mae))
print(sprintf("  R² on test set: %.4f", r2_test))

# Plot predictions vs actual
png("../plots/data_birthwt_predictions.png", width=800, height=600)
plot(birthwt.test$birthwt.grams, birthwt.predict.out,
     main="Predicted vs Actual Birth Weight",
     xlab="Actual Birth Weight (grams)",
     ylab="Predicted Birth Weight (grams)",
     pch=19, col="darkblue")
abline(a=0, b=1, col="red", lwd=2, lty=2)  # Perfect prediction line
# Add regression line
abline(lm(birthwt.predict.out ~ birthwt.test$birthwt.grams), col="blue", lwd=2)
legend("topleft", 
       legend=c("Perfect prediction", "Actual fit", 
                sprintf("R² = %.3f", r2_test)),
       col=c("red", "blue", "white"), 
       lty=c(2, 1, 0), lwd=c(2, 2, 0))
grid(col="gray", lty="dotted")
dev.off()
print("Plot saved: data_birthwt_predictions.png")

# Plot residuals
png("../plots/data_birthwt_test_residuals.png", width=1200, height=400)
par(mfrow=c(1,3))

# Residuals vs predicted
plot(birthwt.predict.out, residuals_test,
     main="Residuals vs Predicted",
     xlab="Predicted Values",
     ylab="Residuals",
     pch=19, col="purple")
abline(h=0, col="red", lwd=2, lty=2)

# Histogram of residuals
hist(residuals_test, breaks=15,
     main="Distribution of Residuals",
     xlab="Residuals",
     col="lightblue")
abline(v=0, col="red", lwd=2, lty=2)

# Q-Q plot
qqnorm(residuals_test, main="Q-Q Plot of Residuals", pch=19, col="darkgreen")
qqline(residuals_test, col="red", lwd=2)

par(mfrow=c(1,1))
dev.off()
print("Plot saved: data_birthwt_test_residuals.png")

# ============================================
# CROSS-VALIDATION
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CROSS-VALIDATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Performing 5-fold cross-validation")

# Simple k-fold cross-validation
k <- 5
set.seed(456)
folds <- cut(seq(1, nrow(birthwt.noout)), breaks=k, labels=FALSE)

cv_r2 <- numeric(k)
cv_rmse <- numeric(k)

for(i in 1:k) {
  test_idx <- which(folds == i)
  train_idx <- which(folds != i)
  
  cv_train <- birthwt.noout[train_idx, ]
  cv_test <- birthwt.noout[test_idx, ]
  
  cv_model <- lm(birthwt.grams ~ . - birthwt.below.2500, data=cv_train)
  cv_pred <- predict(cv_model, newdata=cv_test)
  
  cv_resid <- cv_test$birthwt.grams - cv_pred
  cv_rmse[i] <- sqrt(mean(cv_resid^2))
  cv_r2[i] <- 1 - sum(cv_resid^2) / sum((cv_test$birthwt.grams - mean(cv_test$birthwt.grams))^2)
  
  print(sprintf("Fold %d: RMSE = %.2f, R² = %.4f", i, cv_rmse[i], cv_r2[i]))
}

print(sprintf("\nCross-validation results:"))
print(sprintf("  Mean RMSE: %.2f ± %.2f grams", mean(cv_rmse), sd(cv_rmse)))
print(sprintf("  Mean R²: %.4f ± %.4f", mean(cv_r2), sd(cv_r2)))

# ============================================
# SUMMARY
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Key points from this tutorial:")
print("")
print("✓ Loading and saving R objects is very easy")
print("  - Use save() and load() for .Rda files")
print("  - Use write.csv() and read.csv() for text files")
print("")
print("✓ Reading and writing dataframes is pretty easy")
print("  - read.table() for general text files")
print("  - read.csv() for comma-separated values")
print("  - Many options for customization")
print("")
print("✓ Linear models are very easy via lm()")
print("  - Formula syntax: response ~ predictors")
print("  - summary() gives detailed results")
print("  - Diagnostic plots with plot(model)")
print("")
print("✓ Generalized linear models are pretty easy via glm()")
print("  - Similar syntax to lm()")
print("  - Specify family (binomial, poisson, etc.)")
print("  - Used for binary, count, and other non-normal outcomes")
print("")
print("✓ Model validation is critical")
print("  - Train/test splits for honest evaluation")
print("  - Cross-validation for robust performance estimates")
print("  - Check residuals and diagnostic plots")
print("")
print("✓ For more complex models:")
print("  - Generalized linear mixed models via lme4() and glmm()")
print("  - Hierarchical/multilevel models")
print("  - Random effects and nested structures")

cat("\n")
cat(paste(rep("=", 60), collapse=""), "\n")
cat("DATA AND LINEAR MODELS TUTORIAL COMPLETE\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

final_total_plots <- length(list.files("../plots", pattern="data_.*\\.png"))
print(paste("Total plots generated:", final_total_plots))
print("\nAll plots saved to: ../plots/")
print("\nThank you for completing this tutorial!")
