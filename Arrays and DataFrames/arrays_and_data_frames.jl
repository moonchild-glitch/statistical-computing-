#!/usr/bin/env julia
# Arrays, Matrices, Lists, and DataFrames — Julia translation of the R script

# Required stdlibs
using LinearAlgebra
using Statistics
using Downloads

# Optional packages (guarded): DataFrames, CSV, GLM, Plots
have_DataFrames = false
have_CSV = false
have_GLM = false
have_Plots = false

try
	@eval using DataFrames
	global have_DataFrames = true
catch e
	@warn "DataFrames not available. Data frame sections will be skipped. Install with: ]add DataFrames"
end

try
	@eval using CSV
	global have_CSV = true
catch e
	@warn "CSV not available. CSV reading will be skipped. Install with: ]add CSV"
end

try
	@eval using GLM
	global have_GLM = true
catch e
	@warn "GLM not available. Linear model section will be skipped. Install with: ]add GLM StatsModels"
end

try
	@eval using Plots
	global have_Plots = true
catch e
	@warn "Plots not available. Plotting will be skipped. Install with: ]add Plots"
end

# Ensure a directory exists for saved plots during non-interactive runs
const ROOT = abspath(pwd())
const PLOTS_DIR = joinpath(ROOT, "plots")
isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)

println("# arrays, matrices, lists, and dataframes\n")

# ---- Arrays ----
x = [7, 8, 10, 45]
x_arr = reshape(x, 2, 2)  # column-major, like R
println(x_arr)
println(size(x_arr))
println(ndims(x_arr) == 1) # is.vector equivalent (false)
println(x_arr isa Array)   # is.array equivalent
println(eltype(x_arr))
println(summary(x_arr))

# Access and properties
println(x_arr[1, 2])
println((; shape=size(x_arr), eltype=eltype(x_arr)))
println(x_arr[3])  # linear indexing (column-major)
println(x_arr[1:2, 2])
println(x_arr[:, 2])

# which(x_arr > 9) -> 1-based linear indices (column-major)
mask = x_arr .> 9
lin = findall(vec(mask)) # 1-based linear indices
println(lin)

# Elementwise ops preserve structure
y = -x
y_arr = reshape(y, 2, 2)
println(y_arr .+ x_arr)
println(sum(x_arr, dims=2))  # rowSums

# ---- Example: Price of houses in PA ----
calif_penn_url = "http://www.stat.cmu.edu/~cshalizi/uADA/13/hw/01/calif_penn_2011.csv"

penn = nothing
penn_coefs = nothing
if have_DataFrames && have_CSV
	try
	tmpfile = Downloads.download(calif_penn_url)
	calif_penn = CSV.read(tmpfile, DataFrame)
		global penn = calif_penn[calif_penn.STATEFP .== 42, :]
		if have_GLM
			# Fit linear model Median_house_value ~ Median_household_income
			# Coerce to numeric via pass-through (CSV already numeric in this dataset)
			f = @formula(Median_house_value ~ Median_household_income)
			mdl = lm(f, penn)
			co = coef(mdl)
			# co is Vector with intercept first
			global penn_coefs = Dict("(Intercept)" => co[1], "Median_household_income" => co[2])
			println(penn_coefs)
		else
			@warn "Skipping regression: GLM not installed"
		end
	catch e
		@warn "[Skip] Could not download or parse dataset from $calif_penn_url: $(e)"
	end
else
	@warn "Skipping download: CSV/DataFrames not installed"
end

# Using the constants from the R comments
println(34100 < (-26206.564 + 3.651 * 14719))
println(155900 < (-26206.564 + 3.651 * 48102))

if penn !== nothing && penn_coefs !== nothing && have_Plots
	allegheny_rows = 24:425  # R is inclusive
	if maximum(allegheny_rows) <= nrow(penn)
		medinc = penn[allegheny_rows, :Median_household_income]
		values = penn[allegheny_rows, :Median_house_value]
		fitted = penn_coefs["(Intercept)"] .+ penn_coefs["Median_household_income"] .* medinc

		# Save scatter plot: actual vs predicted
		plt = plot(xlabel="Model-predicted median house values",
				   ylabel="Actual median house values",
				   xlim=(0, 5e5), ylim=(0, 5e5), size=(900, 900))
		scatter!(plt, fitted, values)
		plot!(plt, 0:5_000:500_000, 0:5_000:500_000, color=:grey)
		savefig(plt, joinpath(PLOTS_DIR, "allegheny_actual_vs_predicted.png"))
	end
end

# ---- Matrices ----
factory = [40 1; 60 3]
println(factory isa Array{<:Any, 2})  # is.array(factory)
println(ndims(factory) == 2)           # is.matrix

six_sevens = fill(7, 2, 3)
println(six_sevens)
println(factory * six_sevens)

output = [10, 20]
println(factory * output)
println(output' * factory)  # row vector * matrix

println(transpose(factory))
println(det(factory))
println(diag(factory))

factory_diag_mut = copy(factory)
factory_diag_mut[diagind(factory_diag_mut)] .= [35, 4]
println(factory_diag_mut)

println(diagm(0 => [3, 4]))
println(Matrix{Float64}(I, 2, 2))

factory_inv = inv(factory)
println(factory_inv)
println(factory * factory_inv)

available = [1600, 70]
solution = factory \ available
println(solution)
println(factory * solution)

if have_DataFrames
	factory_df = DataFrame(factory, [:cars, :trucks])
	rownames = ["labor", "steel"]
	println(factory_df)

	available_named = Dict("labor" => 1600, "steel" => 70)
	output_named = Dict("trucks" => 20, "cars" => 10)
	output_vec = [output_named[String(c)] for c in names(factory_df)]
	prod1 = Matrix(factory_df) * output_vec
	println(prod1)
	prod2 = Matrix(factory_df) * [output_named[String(c)] for c in names(factory_df)]
	println(prod2)
	avail_vec = [available_named[r] for r in rownames]
	ok = all((Matrix(factory_df) * output_vec) .<= avail_vec)
	println(ok)

	# Doing the same thing to each row or column
	println(vec(mean(Matrix(factory_df), dims=1)))  # colMeans
	println(describe(factory_df))                   # summary
	println(vec(mean(Matrix(factory_df), dims=2)))  # rowMeans
	println(vec(mean(Matrix(factory_df), dims=2)))
end

# ---- Lists ----
my_distribution = Any["exponential", 7, false]
println(my_distribution)
println(all(el -> el isa String, my_distribution))
println(my_distribution[1] isa String)
println(my_distribution[2]^2)

push!(my_distribution, 7)
println(my_distribution)
println(length(my_distribution))
resize!(my_distribution, 3)
println(my_distribution)

my_distribution_named = Dict("family" => "exponential", "mean" => 7, "is.symmetric" => false)
println(my_distribution_named)
println(my_distribution_named["family"])         # [["family"]]
println(Dict("family" => my_distribution_named["family"])) # ["family"] retaining key
println(my_distribution_named["family"])         # like $family
println(my_distribution_named["family"])         # mimic repeated access

another_distribution = Dict("family" => "gaussian", "mean" => 7, "sd" => 1, "is.symmetric" => true)
my_distribution_named["was.estimated"] = false
my_distribution_named["last.updated"] = "2011-08-30"
pop!(my_distribution_named, "was.estimated", nothing)

# ---- DataFrames basics ----
if have_DataFrames
	a_df = DataFrame(v1 = [35, 10], v2 = [8, 4], logicals = Any[true, false])
	println(a_df)
	println(a_df[:, :v1])  # a.matrix[, "v1"]
	println(a_df)
	println(a_df.v1)
	println(a_df[:, :v1])
	println(a_df[1, :])
	# colMeans over numeric columns only
	numcols = filter(nm -> eltype(a_df[!, nm]) <: Number, names(a_df))
	println(Dict(nm => mean(skipmissing(a_df[!, nm])) for nm in numcols))

	println(vcat(a_df, DataFrame(v1 = -3, v2 = -5, logicals = true)))
	println(vcat(a_df, DataFrame(v1 = 3, v2 = 4, logicals = 6)))

	plan = Dict("factory" => have_DataFrames ? a_df : nothing,
				"available" => available,
				"output" => output)
	println(plan["output"])  # plan$output
end

# ---- Eigenstuff ----
eig = eigen(factory)
println(Dict("values" => eig.values, "vectors" => eig.vectors))
println(typeof(eig))
println(factory * eig.vectors[:, 2])
println(eig.values[2] * eig.vectors[:, 2])
println(eig.values[2])
println(eig.values[2])

# ---- States dataframe example ----
if have_DataFrames && have_Plots
	# Synthetic minimal states-like dataframe to allow operations to run
	states = DataFrame(
		state = ["Alabama", "Alaska", "Arizona", "Arkansas", "California"],
		Population = [3615, 365, 2212, 2110, 21198],
		Income = [3624, 6315, 4530, 3378, 5114],
		Illiteracy = [2.1, 1.5, 1.8, 0.7, 1.1],
		Life_Exp = [69.05, 69.31, 70.55, 70.66, 71.71],
		Murder = [15.1, 11.3, 7.8, 10.1, 10.3],
		HS_Grad = [41.3, 66.7, 58.1, 62.1, 58.8],
		Frost = [20, 152, 15, 65, 20],
		Area = [50708, 566432, 113417, 51945, 156361],
		abb = ["AL", "AK", "AZ", "AR", "CA"],
		region = ["South", "West", "West", "South", "West"],
		division = ["East South Central", "Pacific", "Mountain", "West South Central", "Pacific"],
	)

	println(names(states))
	println(states[1, :])

	# Access patterns similar to R; fallbacks because synthetic dataset has 5 rows
	println(nrow(states) >= 49 ? states[49, 3] : states[end, 3])
	wis = findfirst(==("Wisconsin"), states.state)
	println(wis === nothing ? states[end, :Illiteracy] : states[wis, :Illiteracy])
	println(wis === nothing ? states[end, :] : states[wis, :])
	println(states[1:min(5, nrow(states)), 3])
	println(first(states[:, :Illiteracy], 5))
	println(first(states[:, :Illiteracy], 5))
	println(states[states.division .== "New England", :Illiteracy])
	println(states[states.region .== "South", :Illiteracy])
	println(describe(states[:, :HS_Grad]))

	states.HS_Grad .= states.HS_Grad ./ 100
	println(describe(states[:, :HS_Grad]))
	states.HS_Grad .= 100 .* states.HS_Grad

	# with-like calculation
	calc = 100 .* (states.HS_Grad ./ (100 .- states.Illiteracy))
	println(first(calc, 5))
	println(first(calc, 5))

	# Plot Illiteracy vs Frost
	plt2 = scatter(states.Frost, states.Illiteracy, xlabel="Frost", ylabel="Illiteracy", size=(900, 700))
	savefig(plt2, joinpath(PLOTS_DIR, "illiteracy_vs_frost.png"))
end

println()
println("## SUMMARY")
println("Arrays add multi-dimensional structure to vectors")
println("Matrices act like you'd hope they would")
println("Lists let us combine different types of data")
println("Dataframes are hybrids of matrices and lists, for classic tabular data")
println("Recursion lets us build complicated data structures out of the simpler ones")

