# ============================================
# GRAPHICS IN JULIA
# ============================================
# 
# AGENDA:
# - High-level graphics with Plots.jl
# - Custom graphics
# - Layered graphics with multiple backends
# ============================================

using Plots
using StatsPlots
using Statistics
using StatsBase
using Distributions
using DataFrames
using Random

# Set default backend
gr()

# Create plots directory if it doesn't exist
if !isdir("plots")
    mkdir("plots")
end

# ============================================
# FUNCTIONS FOR GRAPHICS
# ============================================
# Plots.jl provides a unified interface for multiple backends
# StatsPlots adds statistical recipes
# Available backends: GR, PyPlot, PlotlyJS, PGFPlots, etc.
# ============================================

# ============================================
# HIGH-LEVEL GRAPHICS
# ============================================

# ============================================
# UNIVARIATE DATA: HISTOGRAM
# ============================================
# Create state income data
state_income = [3098, 3545, 4354, 3378, 4809, 4091, 5348, 4817, 
                4815, 3694, 4091, 3875, 4842, 3974, 4481, 3897, 
                3605, 3617, 3688, 4167, 4751, 4540, 3834, 3820, 
                4188, 3635, 4205, 4476, 3942, 4563, 4566, 4119, 
                3811, 3977, 4657, 3646, 3815, 4445, 3834, 4167, 
                3907, 4701, 4425, 4139, 4364, 4537, 3795, 3821, 
                4281, 4564]

# Create histogram
p = histogram(state_income, bins=8, 
              color=:lightblue, 
              linecolor=:black,
              xlabel="Income",
              ylabel="Frequency",
              title="Histogram of State Income in 1977",
              legend=false)
savefig(p, "plots/state_income_histogram.png")

println("Histogram saved to plots/state_income_histogram.png")

# ============================================
# UNIVARIATE DATA: HISTOGRAM (EARTHQUAKE DEPTHS)
# ============================================
# Simulate earthquake depth data
Random.seed!(42)
earthquake_depth = rand(Exponential(200), 1000)
earthquake_depth = earthquake_depth[earthquake_depth .< 700]

# Create histogram
p = histogram(earthquake_depth, bins=0:70:700,
              color=:lightcoral,
              linecolor=:black,
              xlabel="Earthquake Depth",
              ylabel="Frequency",
              title="Histogram of Earthquake Depths",
              legend=false)
savefig(p, "plots/earthquake_depth_histogram.png")

println("Histogram saved to plots/earthquake_depth_histogram.png")

# ============================================
# EMPIRICAL CDF
# ============================================
# Plot empirical CDF for state income data
sorted_income = sort(state_income)
cumulative = collect(1:length(sorted_income)) ./ length(sorted_income)

p = plot(sorted_income, cumulative, 
         seriestype=:steppost,
         xlabel="Income",
         ylabel="Cumulative Probability",
         title="ECDF of State Income in 1977",
         legend=false,
         linewidth=2)
savefig(p, "plots/state_income_ecdf.png")

println("Empirical CDF saved to plots/state_income_ecdf.png")

# Plot empirical CDF for earthquake depth data
sorted_depth = sort(earthquake_depth)
cumulative = collect(1:length(sorted_depth)) ./ length(sorted_depth)

p = plot(sorted_depth, cumulative,
         seriestype=:steppost,
         xlabel="Earthquake Depth",
         ylabel="Cumulative Probability",
         title="ECDF of Earthquake Depths",
         legend=false,
         linewidth=2)
savefig(p, "plots/earthquake_depth_ecdf.png")

println("Empirical CDF saved to plots/earthquake_depth_ecdf.png")

# ============================================
# Q-Q PLOTS
# ============================================
# Q-Q plot for state income data
p = qqplot(Normal, state_income,
           xlabel="Theoretical Quantiles",
           ylabel="Sample Quantiles",
           title="Q-Q Plot of State Income (1977)",
           markersize=4,
           legend=false)
savefig(p, "plots/state_income_qqnorm.png")

println("QQ plot saved to plots/state_income_qqnorm.png")

# Q-Q plot for earthquake depth data
p = qqplot(Normal, earthquake_depth,
           xlabel="Theoretical Quantiles",
           ylabel="Sample Quantiles",
           title="Q-Q Plot of Earthquake Depths",
           markersize=4,
           legend=false)
savefig(p, "plots/earthquake_depth_qqnorm.png")

println("QQ plot saved to plots/earthquake_depth_qqnorm.png")

# ============================================
# BOX PLOTS
# ============================================
# Create insect spray data
Random.seed!(42)
spray_data = DataFrame(
    count = vcat(
        rand(Poisson(15), 12),  # Spray A
        rand(Poisson(14), 12),  # Spray B
        rand(Poisson(3), 12),   # Spray C
        rand(Poisson(5), 12),   # Spray D
        rand(Poisson(4), 12),   # Spray E
        rand(Poisson(16), 12)   # Spray F
    ),
    spray = repeat(["A", "B", "C", "D", "E", "F"], inner=12)
)

p = @df spray_data boxplot(:spray, :count,
                           xlabel="Spray",
                           ylabel="Count",
                           title="Insect Counts by Spray Type",
                           fillcolor=:lightgreen,
                           legend=false)
savefig(p, "plots/insect_spray_boxplot.png")

println("Box plot saved to plots/insect_spray_boxplot.png")

# ============================================
# SCATTERPLOTS
# ============================================
# Create earthquake location data
Random.seed!(42)
quake_long = rand(Uniform(165, 185), 1000)
quake_lat = rand(Uniform(-38, -10), 1000)
quake_mag = rand(Uniform(4, 6.5), 1000)

p = scatter(quake_long, quake_lat,
            markersize=3,
            alpha=0.6,
            color=:blue,
            xlabel="Longitude",
            ylabel="Latitude",
            title="Location of Earthquake Epicenters",
            legend=false)
savefig(p, "plots/earthquake_locations_scatterplot.png")

println("Scatterplot saved to plots/earthquake_locations_scatterplot.png")

# Scatterplot with symbols scaled by magnitude
sizes = 10 .^ quake_mag ./ 100
p = scatter(quake_long, quake_lat,
            markersize=sizes,
            alpha=0.3,
            xlabel="Longitude",
            ylabel="Latitude",
            title="Location of Earthquake Epicenters",
            legend=false)
savefig(p, "plots/earthquake_locations_symbols.png")

println("Symbol plot saved to plots/earthquake_locations_symbols.png")

# ============================================
# PAIRS PLOT (SCATTERPLOT MATRIX)
# ============================================
# Create trees dataset
Random.seed!(42)
trees_data = DataFrame(
    Girth = randn(31) .* 3 .+ 13,
    Height = randn(31) .* 6 .+ 76,
    Volume = rand(Gamma(15, 2), 31)
)

p = @df trees_data corrplot([:Girth :Height :Volume],
                            title="Pairs Plot of Tree Measurements",
                            grid=false)
savefig(p, "plots/trees_pairs_plot.png")

println("Pairs plot saved to plots/trees_pairs_plot.png")

# ============================================
# THREE DIMENSIONAL PLOTS
# ============================================
# Create criminal data (height vs finger length)
x = 0:0.5:10
y = 0:0.5:10
z = [sin(xi) * cos(yi) * 10 + randn() for xi in x, yi in y]

# Contour plot
p = contour(x, y, z',
            xlabel="Height",
            ylabel="Finger Length",
            title="Contour Plot of Criminal Data",
            fill=true)
savefig(p, "plots/crimtab_contour_plot.png")

println("Contour plot saved to plots/crimtab_contour_plot.png")

# Heatmap (image plot)
p = heatmap(x, y, z',
            xlabel="Height",
            ylabel="Finger Length",
            title="Image Plot of Criminal Data",
            color=:viridis)
savefig(p, "plots/crimtab_image_plot.png")

println("Image plot saved to plots/crimtab_image_plot.png")

# Surface plot (3D perspective)
p = surface(x, y, z',
            xlabel="Height",
            ylabel="Finger Length",
            zlabel="Frequency",
            title="Perspective Plot of Criminal Data",
            camera=(30, 30))
savefig(p, "plots/crimtab_persp_plot.png")

println("Perspective plot saved to plots/crimtab_persp_plot.png")

# ============================================
# CATEGORICAL DATA: PIE CHARTS
# ============================================
pie_sales = [0.12, 0.30, 0.26, 0.16, 0.04, 0.12]
labels = ["Blueberry", "Cherry", "Apple", "Boston Creme", "Other", "Vanilla Creme"]

p = pie(labels, pie_sales,
        title="Pie Sales Distribution",
        legend=false)
savefig(p, "plots/pie_sales_chart.png")

println("Pie chart saved to plots/pie_sales_chart.png")

# Bar plot
va_deaths = DataFrame(
    age_group = ["50-54", "55-59", "60-64", "65-69", "70-74"],
    rural_male = [11.7, 18.1, 26.9, 41.0, 66.0],
    rural_female = [8.7, 11.7, 20.3, 30.9, 54.3],
    urban_male = [15.4, 24.3, 37.0, 54.6, 71.1],
    urban_female = [8.4, 13.6, 19.3, 35.1, 50.0]
)

p = groupedbar(1:5, 
               Matrix(va_deaths[:, 2:5]),
               xlabel="Age Group",
               ylabel="Death Rate per 1000",
               title="Virginia Death Rates per 1000 in 1940",
               xticks=(1:5, va_deaths.age_group),
               label=["Rural Male" "Rural Female" "Urban Male" "Urban Female"],
               legend=:topleft)
savefig(p, "plots/va_deaths_barplot.png")

println("Bar plot saved to plots/va_deaths_barplot.png")

# ============================================
# TIME SERIES PLOTS
# ============================================
# Create airline passengers time series
months = 1:144
passengers = 100 .+ months .* 1.5 .+ sin.(months .* 2 .* π ./ 12) .* 30

p = plot(months, passengers,
         xlabel="Date",
         ylabel="Passengers (in thousands)",
         title="International Airline Passengers",
         color=:blue,
         linewidth=2,
         legend=false)
savefig(p, "plots/airline_passengers_ts.png")

println("Time series plot saved to plots/airline_passengers_ts.png")

# Presidential approval ratings time series
Random.seed!(42)
quarters = 1:120
approval = 60 .+ cumsum(randn(120)) ./ 10

p = plot(quarters, approval,
         xlabel="Date",
         ylabel="Approval Rating",
         title="Presidential Approval Ratings",
         color=:darkgreen,
         linewidth=2,
         legend=false)
savefig(p, "plots/presidential_approval_ts.png")

println("Time series plot saved to plots/presidential_approval_ts.png")

# ============================================
# BINOMIAL DISTRIBUTION
# ============================================
x_binom = 0:5
y_binom = pdf.(Binomial(5, 0.4), x_binom)

p = bar(x_binom, y_binom,
        xlabel="Value",
        ylabel="Probability",
        title="Binomial Distribution (n=5, p=0.4)",
        color=:darkblue,
        linewidth=3,
        legend=false)
savefig(p, "plots/binomial_distribution_plot.png")

println("Binomial distribution plot saved to plots/binomial_distribution_plot.png")

# ============================================
# NORMAL DISTRIBUTION
# ============================================
x_norm = -3:0.01:3
y_norm = pdf.(Normal(), x_norm)

p = plot(x_norm, y_norm,
         xlabel="x",
         ylabel="f(x)",
         title="Normal Distribution",
         color=:blue,
         linewidth=2,
         legend=false)
savefig(p, "plots/normal_distribution_plot.png")

println("Normal distribution plot saved to plots/normal_distribution_plot.png")

# ============================================
# TWO EMPIRICAL CDFs: COMPARISON
# ============================================
# Simulate Puromycin-like data
Random.seed!(42)
treated = randn(23) .* 25 .+ 140
untreated = randn(23) .* 20 .+ 110

sorted_treated = sort(treated)
sorted_untreated = sort(untreated)
cum_treated = collect(1:length(sorted_treated)) ./ length(sorted_treated)
cum_untreated = collect(1:length(sorted_untreated)) ./ length(sorted_untreated)

p = plot(sorted_treated, cum_treated,
         seriestype=:steppost,
         label="Treated",
         color=:black,
         xlims=(60, 200),
         xlabel="Reaction Rate",
         ylabel="Cumulative Probability",
         title="Treated versus Untreated",
         linewidth=2)
plot!(sorted_untreated, cum_untreated,
      seriestype=:steppost,
      label="Untreated",
      color=:blue,
      linewidth=2)
savefig(p, "plots/puromycin_ecdf_comparison.png")

println("Puromycin ECDF comparison saved to plots/puromycin_ecdf_comparison.png")

# ============================================
# MULTIPLE PLOTS ON ONE SET OF AXES
# ============================================
x_trig = 0:0.01:(2π)
sine = sin.(x_trig)
cosine = cos.(x_trig)

p = plot(x_trig, sine,
         label="sin(x)",
         color=:black,
         linestyle=:solid,
         linewidth=2,
         xlabel="x",
         ylabel="y",
         title="Sine and Cosine Functions")
plot!(x_trig, cosine,
      label="cos(x)",
      color=:black,
      linestyle=:dash,
      linewidth=2)
savefig(p, "plots/sine_cosine_plot.png")

println("Sine and cosine plot saved to plots/sine_cosine_plot.png")

# ============================================
# MULTIPLE FRAME PLOTS
# ============================================
# Create precipitation data
Random.seed!(42)
precip = rand(Gamma(15, 2), 70)

p1 = boxplot([precip], ylabel="Precipitation", title="Box Plot", legend=false)
p2 = histogram(precip, bins=15, title="Histogram", xlabel="Precipitation", 
               color=:lightblue, legend=false)

sorted_precip = sort(precip)
cum_precip = collect(1:length(sorted_precip)) ./ length(sorted_precip)
p3 = plot(sorted_precip, cum_precip, seriestype=:steppost, 
          title="Empirical CDF", legend=false)

p4 = qqplot(Normal, precip, title="Q-Q Plot", markersize=4, legend=false)

p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 1000))
savefig(p, "plots/precipitation_multiplot.png")

println("Multiple frame plot saved to plots/precipitation_multiplot.png")

# ============================================
# STATISTICAL MODEL PLOT
# ============================================
# Create Puromycin-like data
Random.seed!(42)
conc = repeat([0.02, 0.06, 0.11, 0.22, 0.56, 1.10], 2)
rate = vcat(
    [47, 97, 123, 152, 191, 200] .+ randn(6) .* 5,  # Untreated
    [76, 107, 139, 159, 201, 207] .+ randn(6) .* 5  # Treated
)
state = repeat(["Untreated", "Treated"], inner=6)

puromycin_data = DataFrame(conc=conc, rate=rate, state=state)

p = @df puromycin_data scatter(:conc, :rate, 
                               group=:state,
                               marker=[:circle :square],
                               markersize=8,
                               xlabel="Substrate Concentration",
                               ylabel="Reaction Rate",
                               title="Enzyme Reaction Rate vs Substrate Concentration")
savefig(p, "plots/puromycin_model_plot.png")

println("Statistical model plot saved to plots/puromycin_model_plot.png")

# ============================================
# 3D SURFACE: SINC FUNCTION
# ============================================
x = -8:0.16:8
y = -8:0.16:8

sinc_func(x, y) = begin
    r = sqrt(x^2 + y^2)
    r == 0 ? 1.0 : sin(r) / r
end

z = [sinc_func(xi, yi) for xi in x, yi in y]

p = surface(x, y, z,
            title="3D Surface: sin(r)/r",
            camera=(30, 30),
            colorbar=false)
savefig(p, "plots/persp_wire_mesh.png")

println("3D wire mesh plot saved to plots/persp_wire_mesh.png")

# ============================================
# CUSTOM GRAPHICS: LEEMIS CHAPTER 21
# ============================================
# Note: Julia's Plots.jl has limited low-level plotting capabilities
# For complex custom plots, consider using Luxor.jl or Makie.jl

p = plot(xlims=(0, 9), ylims=(-10, 20),
         framestyle=:none,
         legend=false,
         size=(800, 600))

# Add text annotations
annotate!(2, 18, text("bold font", :left, 10, :bold))
annotate!(2, 16, text("italics font", :left, 10, :italic))
annotate!(2, 14, text("bold & italics font", :left, 10, (:bold, :italic)))
annotate!(2, 12, text("αβΓ", :left, 12))

# Right/top justified
annotate!(7.5, 15, text("right/top justified", :left, 10))
scatter!([7.5], [15], marker=:+, markersize=10)

# Left/bottom justified
annotate!(2, -9, text("left/bottom justified", :left, 10))

# Add curves
x_curve = 0:0.1:8
plot!(x_curve, x_curve.^2 .- 5 .* x_curve .+ 2, color=:black, linewidth=2)
plot!(x_curve, -0.5 .* x_curve .+ 3, color=:black, linestyle=:dash, linewidth=2)

# Slanted text (rotation limited in Plots.jl)
annotate!(2.5, -2, text("slanted Text", :left, 10))

# Polygon (yellow triangle)
plot!(Shape([0, 1, 0], [-3, 0, 2]), color=:yellow, linecolor=:black)

# Plotting region label
annotate!(5, 9, text("plotting region", :center, 10))
plot!([4.5, 3], [9, 7], arrow=true, color=:black)

# Mathematical expression
annotate!(6, -5, text("λᵢ/2ˣ", :center, 14))

# Add points
scatter!([6], [3], markersize=10, markerstrokewidth=2, 
         markercolor=:white, markerstrokecolor=:black)
scatter!([4], [-9], markersize=7, markerstrokewidth=2,
         markercolor=:white, markerstrokecolor=:black)

# Add text
annotate!(7.5, 3, text("w", :center, 12, :blue, :bold))
annotate!(8.5, 3, text("8", :center, 10))

# Add labels
annotate!(4.5, -12, text("c(0, 9) / margin 1", :center, 10))

savefig(p, "plots/leemis_custom_plot.png")

println("Leemis custom plot saved to plots/leemis_custom_plot.png")

# ============================================
# LAYERED GRAPHICS (GGPLOT-STYLE)
# ============================================
println("\n=== Note: ggplot2-style layered graphics ===")
println("For ggplot2-style graphics in Julia, consider:")
println("  - StatsPlots.jl (statistical recipes)")
println("  - Gadfly.jl (Julia's ggplot2)")
println("  - Makie.jl (powerful visualization)")

# Create FEV-like data
Random.seed!(42)
n_samples = 654
fev_data = DataFrame(
    age = rand(3:19, n_samples),
    fev = randn(n_samples) .* 0.8 .+ 2.5,
    height = randn(n_samples) .* 10 .+ 65,
    smoke = rand([0, 1], n_samples),
    sex = rand([0, 1], n_samples)
)

# Adjust fev based on age and smoking
fev_data.fev = 1.0 .+ fev_data.age .* 0.15 .+ randn(n_samples) .* 0.3
fev_data[fev_data.smoke .== 1, :fev] .-= 0.2

println("\n=== FEV Data Structure ===")
println(describe(fev_data))

# ============================================
# LAYERING
# ============================================
# Calculate mean fev for each age group
fev_mean = combine(groupby(fev_data, :age), :fev => mean => :fev)

p = @df fev_data scatter(:age, :fev, 
                         alpha=0.5,
                         markersize=3,
                         label="Observations",
                         xlabel="Age",
                         ylabel="FEV",
                         title="FEV vs Age with Mean Line")
@df fev_mean plot!(:age, :fev, 
                   color=:red,
                   linewidth=2,
                   label="Mean")
savefig(p, "plots/fev_layered_plot.png")

println("FEV layered plot saved to plots/fev_layered_plot.png")

# ============================================
# SMOOTHING
# ============================================
# Scatter plot with LOWESS smooth
p = @df fev_data scatter(:age, :fev,
                         alpha=0.5,
                         markersize=3,
                         label="Data",
                         xlabel="Age",
                         ylabel="FEV",
                         title="FEV vs Age with LOWESS Smooth")

# Add smooth line using moving average as approximation
age_sorted = sort(unique(fev_data.age))
fev_smooth = [mean(fev_data[fev_data.age .== a, :fev]) for a in age_sorted]
plot!(age_sorted, fev_smooth, 
      color=:blue,
      linewidth=3,
      label="Smooth")
savefig(p, "plots/fev_smooth_default.png")

println("FEV smooth (default) saved to plots/fev_smooth_default.png")

# ============================================
# GROUPING
# ============================================
# Group by smoke status
p = @df fev_data scatter(:age, :fev,
                         group=:smoke,
                         alpha=0.5,
                         markersize=3,
                         xlabel="Age",
                         ylabel="FEV",
                         title="FEV vs Age by Smoking Status",
                         label=["Non-smoker" "Smoker"])
savefig(p, "plots/fev_grouped_smoke_color.png")

println("FEV grouped by smoke (colored) saved to plots/fev_grouped_smoke_color.png")

# ============================================
# FACETING
# ============================================
# Create age groups
fev_data.age_group = cut(fev_data.age, 5)

# Create faceted plot (simplified version)
p1 = @df fev_data[fev_data.sex .== 0, :] scatter(:height, :fev,
                                                  group=:smoke,
                                                  alpha=0.6,
                                                  markersize=3,
                                                  title="Sex = 0",
                                                  legend=false)

p2 = @df fev_data[fev_data.sex .== 1, :] scatter(:height, :fev,
                                                  group=:smoke,
                                                  alpha=0.6,
                                                  markersize=3,
                                                  title="Sex = 1",
                                                  legend=:topright)

p = plot(p1, p2, layout=(2, 1), size=(800, 1000))
savefig(p, "plots/fev_faceted_sex_age_group.png")

println("FEV faceted plot saved to plots/fev_faceted_sex_age_group.png")

# ============================================
# SUMMARY
# ============================================
println("\n=== SUMMARY ===")
println("Julia has strong graphic capabilities through Plots.jl, Makie.jl, and Gadfly.jl")
println("Graphing is an iterative process; Don't rely on the default options")
println("Avoid gimmicks, use the minimum amount of ink to get your point across")
println("A small table can be better than a large graph")
println("Carefully consider the size and shape of your graph, bigger is not always better")

println("\n=== Graphics Tutorial Complete ===")
println("All plots saved to plots/ directory")
