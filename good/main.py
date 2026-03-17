import kf_mine

system = kf_mine.System(0.05, 1.0, [0.0, 1.0, 0.0], 0.2, 0.1)

filter = kf_mine.AlphaBetaGammaFilter(0.05, [0.06, 0.05, 0.05], system, system.real_state.copy(), improved=False)

experiment = kf_mine.Experiment(system, filter)

animation = kf_mine.Animator(experiment)

animation.animate(5000)

kf_mine.Plotter.plot_experiment(experiment)