class Utils:
    all_markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h', 'v', '^', '<', '>', '1',
                   '2', '3', '4', '|', '_']
    all_colors = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']

    all_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    @staticmethod
    def assign_markers_and_colors(no_clusters):  # for scatter plot
        markers_and_colors = []
        for i in range(no_clusters):
            markers_and_colors.append(
                (i, Utils.all_markers[i % len(Utils.all_markers)], Utils.all_colors[i % len(Utils.all_colors)]))
        return markers_and_colors

    @staticmethod
    def assign_symbols(samples_with_clusters):  # for scatter plot
        samples_with_symbols = []
        for (sample, cluster) in samples_with_clusters:
            samples_with_symbols.append(
                (sample, Utils.all_symbols[cluster]))
        return samples_with_symbols

    @staticmethod
    def get_colors_array(samples, som):
        colors_array = []
        for s in samples:
            bmu = som.find_BMU(s)
            colors_array.append([float(bmu[0]) / float(som.getX()), float(bmu[1])/ float(som.getY()), float(bmu[2]) / float(som.getZ())])
        return colors_array
