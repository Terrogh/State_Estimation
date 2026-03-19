import kf_mine

system = kf_mine.System(0.05, 1.0, [0.0, 1.0, 0.0], 0.01, 0.01)

filter = kf_mine.AlphaBetaGammaFilter(0.05, [0.1, 0.9, 18.09], system, [0.0, 10.0, 1000.0])

experiment = kf_mine.Experiment(system, filter)

animation = kf_mine.Animator(experiment)

animation.animate(5000)

kf_mine.Plotter.plot_experiment(experiment)