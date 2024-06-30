from tabulate import tabulate

class ModelComparator:
    def __init__(self, results):
        self.results = results
        self.metrics = self._get_all_metrics()
        self.max_values = self._calculate_max_values()

    def _get_all_metrics(self):
        metrics = set()
        for model_results in self.results.values():
            metrics.update(model_results['metrics'].keys())
        return list(metrics)

    def _calculate_max_values(self):
        return {
            'time': max(model['inference_time'] for model in self.results.values()),
            'size': max(model['size'] for model in self.results.values()),
            'metrics': {metric: max(model['metrics'].get(metric, 0) for model in self.results.values()) 
                        for metric in self.metrics}
        }

    def _format_value_with_ratio(self, value, max_value, is_time_or_size=False):
        if value == 'N/A':
            return 'N/A'
        ratio = max_value / value if value != 0 else float('inf')
        if is_time_or_size:
            return f"{value:.2f}", f"{ratio:.2f}x"
        return f"{value:.4f} ({ratio:.2f}x)"

    def _create_row(self, model, model_results):
        time_value, time_ratio = self._format_value_with_ratio(model_results['inference_time'], self.max_values['time'], True)
        size_value, size_ratio = self._format_value_with_ratio(model_results['size'], self.max_values['size'], True)
        
        row = [model, time_value, time_ratio, size_value, size_ratio]
        
        for metric in self.metrics:
            value = model_results['metrics'].get(metric, 'N/A')
            row.append(self._format_value_with_ratio(value, self.max_values['metrics'][metric]))
        
        return row

    def print_comparison(self):
        """
        Print a formatted comparison table of the benchmarking results using tabulate,
        including relative ratios for all columns.
        """
        headers = ['Model', 'Inference Time (ms)', 'Time Ratio', 'Size (KB)', 'Size Ratio'] + self.metrics
        data = [self._create_row(model, model_results) for model, model_results in self.results.items()]
        print(tabulate(data, headers=headers, tablefmt="pretty"))
