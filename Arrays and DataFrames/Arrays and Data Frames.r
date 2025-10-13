#arrays, matrices, lists, and dataframes

# Ensure a directory exists for saved plots during non-interactive runs
plots_dir <- file.path(getwd(), "plots")
if (!dir.exists(plots_dir)) dir.create(plots_dir, recursive = TRUE)

x <- c(7, 8, 10, 45)
x.arr <- array(x,dim=c(2,2))
x.arr
dim(x.arr)
is.vector(x.arr)
is.array(x.arr)
typeof(x.arr)
str(x.arr)

#accessing and operating on arrays
x.arr[1,2]
attributes(x.arr)
x.arr[3]

x.arr[c(1:2),2]
x.arr[,2]

#Functions on arrays
which(x.arr > 9)
#Many functions do preserve array structure:
y <- -x
y.arr <- array(y,dim=c(2,2))
y.arr + x.arr
#Others specifically act on each row or column of the array separately:
rowSums(x.arr)

#Example: Price of houses in PA

# Read data with a guard for network errors (so the rest of the script can still run)
calif_penn_url <- "http://www.stat.cmu.edu/~cshalizi/uADA/13/hw/01/calif_penn_2011.csv"
calif_penn <- tryCatch(
     read.csv(calif_penn_url),
     error = function(e) {
          message("[Skip] Could not download dataset from ", calif_penn_url, ": ", conditionMessage(e))
          NULL
     }
)
if (!is.null(calif_penn)) {
     penn <- calif_penn[calif_penn[,"STATEFP"]==42,]
     coefficients(lm(Median_house_value ~ Median_household_income, data=penn))
}

#Fit a simple linear model, predicting median house price from median household income
#Census tracts 24–425 are Allegheny county
#Tract 24 has a median income of $14,719; actual median house value is $34,100 — is that above or below what's?
34100 < -26206.564 + 3.651*14719
#Tract 25 has income $48,102 and house price $155,900
155900 < -26206.564 + 3.651*48102
#Use variables and names

if (!is.null(calif_penn)) {
     penn.coefs <- coefficients(lm(Median_house_value ~ Median_household_income, data=penn))
     penn.coefs
}


if (!is.null(calif_penn)) {
  allegheny.rows <- 24:425
  allegheny.medinc <- penn[allegheny.rows,"Median_household_income"]
  allegheny.values <- penn[allegheny.rows,"Median_house_value"]
  allegheny.fitted <- penn.coefs["(Intercept)"]+penn.coefs["Median_household_income"]*allegheny.medinc

  # Save scatter plot of actual vs predicted to file
  png(file.path(plots_dir, "allegheny_actual_vs_predicted.png"), width = 900, height = 900, res = 120)
  plot(x=allegheny.fitted, y=allegheny.values,
       xlab="Model-predicted median house values",
       ylab="Actual median house values",
       xlim=c(0,5e5),ylim=c(0,5e5))
  abline(a=0,b=1,col="grey")
  dev.off()
}


##Matrices
factory <- matrix(c(40,1,60,3),nrow=2)
is.array(factory)

is.matrix(factory)

#Matrix multiplication
six.sevens <- matrix(rep(7,6),ncol=3)
six.sevens

factory %*% six.sevens # [2x2] * [2x3]


#Multiplying matrices and vectors
output <- c(10,20)
factory %*% output

output %*% factory

#R silently casts the vector as either a row or a column matrix

#Matrix operators

t(factory)

det(factory)

#The diagonal

diag(factory)

diag(factory) <- c(35,4)
factory
diag(factory) <- c(40,3)

#Creating a diagonal or identity matrix

diag(c(3,4))

diag(2)

#Inverting a matrix

solve(factory)

factory %*% solve(factory)

##Why's it called "solve"" anyway?

available <- c(1600,70)
solve(factory,available)

factory %*% solve(factory,available)

#Names in matrices

rownames(factory) <- c("labor","steel")
colnames(factory) <- c("cars","trucks")
factory

available <- c(1600,70)
names(available) <- c("labor","steel")

output <- c(20,10)
names(output) <- c("trucks","cars")
factory %*% output # But we've got cars and trucks mixed up!

factory %*% output[colnames(factory)]

all(factory %*% output[colnames(factory)] <= available[rownames(factory)])


#Doing the same thing to each row or column

colMeans(factory)

summary(factory)

rowMeans(factory)

apply(factory,1,mean)


#LISTS
my.distribution <- list("exponential",7,FALSE)
my.distribution

#Accessing pieces of lists
is.character(my.distribution)

is.character(my.distribution[[1]])

my.distribution[[2]]^2

#Expanding and contracting lists

my.distribution <- c(my.distribution,7)
my.distribution

length(my.distribution)

length(my.distribution) <- 3
my.distribution


#Naming list elements

names(my.distribution) <- c("family","mean","is.symmetric")
my.distribution

my.distribution[["family"]]

my.distribution["family"]

#Lists have a special short-cut way of using names, $ (which removes names and structures):

my.distribution[["family"]]

my.distribution$family

#Names in lists

another.distribution <- list(family="gaussian",mean=7,sd=1,is.symmetric=TRUE)

my.distribution$was.estimated <- FALSE
my.distribution[["last.updated"]] <- "2011-08-30"

my.distribution$was.estimated <- NULL

#Key-Value pairs
#Lists give us a way to store and look up data by name, rather than by position
#Dataframes
#Dataframe = the classic data table, nn rows for cases, pp columns for variables
#Lots of the really-statistical parts of R presume data frames penn from last time was really a dataframe


a.matrix <- matrix(c(35,8,10,4),nrow=2)
colnames(a.matrix) <- c("v1","v2")
a.matrix

a.matrix[,"v1"]  # Try a.matrix$v1 and see what happens

a.data.frame <- data.frame(a.matrix,logicals=c(TRUE,FALSE))
a.data.frame

a.data.frame$v1

a.data.frame[,"v1"]

a.data.frame[1,]

colMeans(a.data.frame)


#Adding rows and columns

rbind(a.data.frame,list(v1=-3,v2=-5,logicals=TRUE))

rbind(a.data.frame,c(3,4,6))

#Structures of Structures

plan <- list(factory=factory, available=available, output=output)
plan$output

#Example: Eigenstuff

eigen(factory)

class(eigen(factory))

factory %*% eigen(factory)$vectors[,2]

eigen(factory)$values[2] * eigen(factory)$vectors[,2]

eigen(factory)$values[2]

eigen(factory)[[1]][[2]]


#Creating an example dataframe

library(datasets)
states <- data.frame(state.x77, abb=state.abb, region=state.region, division=state.division)

colnames(states)

states[1,]

#Dataframe access

states[49,3]

states["Wisconsin","Illiteracy"]

states["Wisconsin",]

head(states[,3])

head(states[,"Illiteracy"])

head(states$Illiteracy)

states[states$division=="New England", "Illiteracy"]

states[states$region=="South", "Illiteracy"]

summary(states$HS.Grad)


states$HS.Grad <- states$HS.Grad/100
summary(states$HS.Grad)


states$HS.Grad <- 100*states$HS.Grad


#with()
#What percentage of literate adults graduated HS?
head(100*(states$HS.Grad/(100-states$Illiteracy)))

#with() takes a data frame and evaluates an expression "inside" it:
with(states, head(100*(HS.Grad/(100-Illiteracy))))


#Data arguments

# Save plot of Illiteracy vs Frost to file for non-interactive runs
png(file.path(plots_dir, "illiteracy_vs_frost.png"), width = 900, height = 700, res = 120)
plot(Illiteracy~Frost, data=states)
dev.off()


##SUMMARY
#Arrays add multi-dimensional structure to vectors
#Matrices act like you'd hope they would
#Lists let us combine different types of data
#Dataframes are hybrids of matrices and lists, for classic tabular data
#Recursion lets us build complicated data structures out of the simpler ones