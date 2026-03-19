import kf_mine

system = kf_mine.System(0.05, 1.0, [0.0, 1.0, 0.0], 0.2, 0.1)

filter = kf_mine.AlphaBetaGammaFilter(0.05, [0.2, 0.2, 0.2], system, system.real_state.copy(), improved=True)

experiment = kf_mine.Experiment(system, filter)

# experiment.run(2000)
# kf_mine.Plotter.plot_experiment(experiment)

def RMS(arr):
    square_sum = sum(v**2 for v in arr)
    return (square_sum / len(arr))**0.5 

print(f"RMS(RMSE): {RMS(experiment.multiplerun(10, 2000)):.4f}")