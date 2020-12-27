SCENARIOS = ["demand",
             "wind",
             "solar",]
             # "wind_elevation",
             # "demand_elevation",
             # "solar_elevation",
             # "wind_wettemp",
             # "solar_wettemp",
             # "demand_wettemp",
             # "wind_drytemp",
             # "solar_drytemp",
             # "demand_drytemp",
             # "wind_pressure",
             # "solar_pressure",
             # "demand_pressure",
             # "wind_humidity",
             # "solar_humidity",
             # "demand_humidity",
             # "wind_windspeed",
             # "solar_windspeed",
             # "demand_windspeed"]

HOURS = ['04', '48']

rule all:
    input:
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
