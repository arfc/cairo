SCENARIOS = ["demand",
             "wind",
             "solar",
             "wind_elevation",
             "demand_elevation",
             "solar_elevation",
             "wind_wettemp",
             "solar_wettemp",
             "demand_wettemp",
             "wind_drytemp",
             "solar_drytemp",
             "demand_drytemp",
             "wind_pressure",
             "solar_pressure",
             "demand_pressure",
             "wind_humidity",
             "solar_humidity",
             "demand_humidity",
             "wind_windspeed",
             "solar_windspeed",
             "demand_windspeed"]

HOURS = ['04', '48']

rule all:
    input:
        "data/lorenz63_rho_noise_loss.npy",
        "data/lorenz63_n_reservoir_sparsity_loss.npy",
        "data/lorenz63_trainlen_loss.npy",
        "data/lorenz63_prediction.npy",
        "figures/lorenz63_prediction.png",
        expand("data/48_{scenario}_rho_noise_loss.npy", scenario=SCENARIOS),
        expand("data/48_{scenario}_n_reservoir_sparsity_loss.npy", scenario=SCENARIOS),
        expand("data/48_{scenario}_trainlen_loss.npy", scenario=SCENARIOS),
        expand("data/48_{scenario}_prediction.npy", scenario=SCENARIOS),
        expand("figures/48_{scenario}_prediction.png", scenario=SCENARIOS),
        # expand("data/04_{scenario}_rho_noise_loss.npy", scenario=SCENARIOS),
        # expand("data/04_{scenario}_n_reservoir_sparsity_loss.npy", scenario=SCENARIOS),
        # expand("data/04_{scenario}_trainlen_loss.npy", scenario=SCENARIOS),
        # expand("data/04_{scenario}_prediction.npy", scenario=SCENARIOS),
        # expand("figures/04_{scenario}_prediction.png", scenario=SCENARIOS),

rule optimize_lorenz_63:
    input: "lorenz.py"
    output:
        "data/lorenz63_rho_noise_loss.npy",
        "data/lorenz63_n_reservoir_sparsity_loss.npy",
        "data/lorenz63_trainlen_loss.npy",
        "data/lorenz63_prediction.npy",
        "figures/lorenz63_prediction.png",
    shell:
        "python lorenz_driver.py -L -S lorenz63"

# ============================================================================
# Standard predictions
# ============================================================================

rule optimize_demand_48:
    input: "data/uiuc_demand_data.csv"
    output:
        "data/48_demand_rho_noise_loss.npy",
        "data/48_demand_n_reservoir_sparsity_loss.npy",
        "data/48_demand_trainlen_loss.npy",
        "data/48_demand_prediction.npy",
        "figures/48_demand_prediction.png",
    shell:
        "python driver.py -i {input} -H 48 -S 48_demand"

rule optimize_wind_48:
    input: "data/railsplitter_data.csv"
    output:
        "data/48_wind_rho_noise_loss.npy",
        "data/48_wind_n_reservoir_sparsity_loss.npy",
        "data/48_wind_trainlen_loss.npy",
        "data/48_wind_prediction.npy",
        "figures/48_wind_prediction.png",
    shell:
        "python driver.py -i {input} -H 48 -S 48_wind"

rule optimize_solar_48:
    input: "data/solarfarm_data.csv"
    output:
        "data/48_solar_rho_noise_loss.npy",
        "data/48_solar_n_reservoir_sparsity_loss.npy",
        "data/48_solar_trainlen_loss.npy",
        "data/48_solar_prediction.npy",
        "figures/48_solar_prediction.png",
    shell:
        "python driver.py -i {input} -H 48 -S 48_solar"

# ============================================================================
# Predictions with Solar Elevation
# ============================================================================
rule optimize_demand_elevation_48:
    input: "data/uiuc_demand_data.csv"
    output:
        "data/48_demand_elevation_rho_noise_loss.npy",
        "data/48_demand_elevation_n_reservoir_sparsity_loss.npy",
        "data/48_demand_elevation_trainlen_loss.npy",
        "data/48_demand_elevation_prediction.npy",
        "figures/48_demand_elevation_prediction.png",
    shell:
        "python driver.py -i {input} -e -H 48 -S 48_demand_elevation"

rule optimize_wind_elevation_48:
    input: "data/railsplitter_data.csv"
    output:
        "data/48_wind_elevation_rho_noise_loss.npy",
        "data/48_wind_elevation_n_reservoir_sparsity_loss.npy",
        "data/48_wind_elevation_trainlen_loss.npy",
        "data/48_wind_elevation_prediction.npy",
        "figures/48_wind_elevation_prediction.png",
    shell:
        "python driver.py -i {input} -e -H 48 -S 48_wind_elevation"

rule optimize_solar_elevation_48:
    input: "data/solarfarm_data.csv"
    output:
        "data/48_solar_elevation_rho_noise_loss.npy",
        "data/48_solar_elevation_n_reservoir_sparsity_loss.npy",
        "data/48_solar_elevation_trainlen_loss.npy",
        "data/48_solar_elevation_prediction.npy",
        "figures/48_solar_elevation_prediction.png",
    shell:
        "python driver.py -i {input} -e -H 48 -S 48_solar_elevation"

# ============================================================================
# Predictions with drytemp
# ============================================================================
rule optimize_demand_drytemp_48:
    input:
        demand = "data/uiuc_demand_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_demand_drytemp_rho_noise_loss.npy",
        "data/48_demand_drytemp_n_reservoir_sparsity_loss.npy",
        "data/48_demand_drytemp_trainlen_loss.npy",
        "data/48_demand_drytemp_prediction.npy",
        "figures/48_demand_drytemp_prediction.png",
    shell:
        "python driver.py -i {input.demand} -f {input.weather} -d -H 48 -S 48_demand_drytemp"

rule optimize_wind_drytemp_48:
    input:
        wind = "data/railsplitter_data.csv",
        weather = "data/lincoln_weatherdata.csv"
    output:
        "data/48_wind_drytemp_rho_noise_loss.npy",
        "data/48_wind_drytemp_n_reservoir_sparsity_loss.npy",
        "data/48_wind_drytemp_trainlen_loss.npy",
        "data/48_wind_drytemp_prediction.npy",
        "figures/48_wind_drytemp_prediction.png",
    shell:
        "python driver.py -i {input.wind} -f {input.weather} -d -H 48 -S 48_wind_drytemp"

rule optimize_solar_drytemp_48:
    input:
        solar = "data/solarfarm_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_solar_drytemp_rho_noise_loss.npy",
        "data/48_solar_drytemp_n_reservoir_sparsity_loss.npy",
        "data/48_solar_drytemp_trainlen_loss.npy",
        "data/48_solar_drytemp_prediction.npy",
        "figures/48_solar_drytemp_prediction.png",
    shell:
        "python driver.py -i {input.solar} -f {input.weather} -d -H 48 -S 48_solar_drytemp"

# ============================================================================
# Predictions with wettemp
# ============================================================================
rule optimize_demand_wettemp_48:
    input:
        demand = "data/uiuc_demand_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_demand_wettemp_rho_noise_loss.npy",
        "data/48_demand_wettemp_n_reservoir_sparsity_loss.npy",
        "data/48_demand_wettemp_trainlen_loss.npy",
        "data/48_demand_wettemp_prediction.npy",
        "figures/48_demand_wettemp_prediction.png",
    shell:
        "python driver.py -i {input.demand} -f {input.weather} -w -H 48 -S 48_demand_wettemp"

rule optimize_wind_wettemp_48:
    input:
        wind = "data/railsplitter_data.csv",
        weather = "data/lincoln_weatherdata.csv"
    output:
        "data/48_wind_wettemp_rho_noise_loss.npy",
        "data/48_wind_wettemp_n_reservoir_sparsity_loss.npy",
        "data/48_wind_wettemp_trainlen_loss.npy",
        "data/48_wind_wettemp_prediction.npy",
        "figures/48_wind_wettemp_prediction.png",
    shell:
        "python driver.py -i {input.wind} -f {input.weather} -w -H 48 -S 48_wind_wettemp"

rule optimize_solar_wettemp_48:
    input:
        solar = "data/solarfarm_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_solar_wettemp_rho_noise_loss.npy",
        "data/48_solar_wettemp_n_reservoir_sparsity_loss.npy",
        "data/48_solar_wettemp_trainlen_loss.npy",
        "data/48_solar_wettemp_prediction.npy",
        "figures/48_solar_wettemp_prediction.png",
    shell:
        "python driver.py -i {input.solar} -f {input.weather} -w -H 48 -S 48_solar_wettemp"

# ============================================================================
# Predictions with pressure
# ============================================================================
rule optimize_demand_pressure_48:
    input:
        demand = "data/uiuc_demand_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_demand_pressure_rho_noise_loss.npy",
        "data/48_demand_pressure_n_reservoir_sparsity_loss.npy",
        "data/48_demand_pressure_trainlen_loss.npy",
        "data/48_demand_pressure_prediction.npy",
        "figures/48_demand_pressure_prediction.png",
    shell:
        "python driver.py -i {input.demand} -f {input.weather} -p -H 48 -S 48_demand_pressure"

rule optimize_wind_pressure_48:
    input:
        wind = "data/railsplitter_data.csv",
        weather = "data/lincoln_weatherdata.csv"
    output:
        "data/48_wind_pressure_rho_noise_loss.npy",
        "data/48_wind_pressure_n_reservoir_sparsity_loss.npy",
        "data/48_wind_pressure_trainlen_loss.npy",
        "data/48_wind_pressure_prediction.npy",
        "figures/48_wind_pressure_prediction.png",
    shell:
        "python driver.py -i {input.wind} -f {input.weather} -p -H 48 -S 48_wind_pressure"

rule optimize_solar_pressure_48:
    input:
        solar = "data/solarfarm_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_solar_pressure_rho_noise_loss.npy",
        "data/48_solar_pressure_n_reservoir_sparsity_loss.npy",
        "data/48_solar_pressure_trainlen_loss.npy",
        "data/48_solar_pressure_prediction.npy",
        "figures/48_solar_pressure_prediction.png",
    shell:
        "python driver.py -i {input.solar} -f {input.weather} -p -H 48 -S 48_solar_pressure"

# ============================================================================
# Predictions with humidity
# ============================================================================
rule optimize_demand_humidity_48:
    input:
        demand = "data/uiuc_demand_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_demand_humidity_rho_noise_loss.npy",
        "data/48_demand_humidity_n_reservoir_sparsity_loss.npy",
        "data/48_demand_humidity_trainlen_loss.npy",
        "data/48_demand_humidity_prediction.npy",
        "figures/48_demand_humidity_prediction.png",
    shell:
        "python driver.py -i {input.demand} -f {input.weather} -h -H 48 -S 48_demand_humidity"

rule optimize_wind_humidity_48:
    input:
        wind = "data/railsplitter_data.csv",
        weather = "data/lincoln_weatherdata.csv"
    output:
        "data/48_wind_humidity_rho_noise_loss.npy",
        "data/48_wind_humidity_n_reservoir_sparsity_loss.npy",
        "data/48_wind_humidity_trainlen_loss.npy",
        "data/48_wind_humidity_prediction.npy",
        "figures/48_wind_humidity_prediction.png",
    shell:
        "python driver.py -i {input.wind} -f {input.weather} -h -H 48 -S 48_wind_humidity"

rule optimize_solar_humidity_48:
    input:
        solar = "data/solarfarm_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_solar_humidity_rho_noise_loss.npy",
        "data/48_solar_humidity_n_reservoir_sparsity_loss.npy",
        "data/48_solar_humidity_trainlen_loss.npy",
        "data/48_solar_humidity_prediction.npy",
        "figures/48_solar_humidity_prediction.png",
    shell:
        "python driver.py -i {input.solar} -f {input.weather} -h -H 48 -S 48_solar_humidity"

# ============================================================================
# Predictions with windspeed
# ============================================================================
rule optimize_demand_windspeed_48:
    input:
        demand = "data/uiuc_demand_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_demand_windspeed_rho_noise_loss.npy",
        "data/48_demand_windspeed_n_reservoir_sparsity_loss.npy",
        "data/48_demand_windspeed_trainlen_loss.npy",
        "data/48_demand_windspeed_prediction.npy",
        "figures/48_demand_windspeed_prediction.png",
    shell:
        "python driver.py -i {input.demand} -f {input.weather} -u -H 48 -S 48_demand_windspeed"

rule optimize_wind_windspeed_48:
    input:
        wind = "data/railsplitter_data.csv",
        weather = "data/lincoln_weatherdata.csv"
    output:
        "data/48_wind_windspeed_rho_noise_loss.npy",
        "data/48_wind_windspeed_n_reservoir_sparsity_loss.npy",
        "data/48_wind_windspeed_trainlen_loss.npy",
        "data/48_wind_windspeed_prediction.npy",
        "figures/48_wind_windspeed_prediction.png",
    shell:
        "python driver.py -i {input.wind} -f {input.weather} -u -H 48 -S 48_wind_windspeed"

rule optimize_solar_windspeed_48:
    input:
        solar = "data/solarfarm_data.csv",
        weather = "data/champaign_weatherdata.csv"
    output:
        "data/48_solar_windspeed_rho_noise_loss.npy",
        "data/48_solar_windspeed_n_reservoir_sparsity_loss.npy",
        "data/48_solar_windspeed_trainlen_loss.npy",
        "data/48_solar_windspeed_prediction.npy",
        "figures/48_solar_windspeed_prediction.png",
    shell:
        "python driver.py -i {input.solar} -f {input.weather} -u -H 48 -S 48_solar_windspeed"
