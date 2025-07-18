from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
       
        min_per_day = self.data.min(dim=1).values
        max_per_day = self.data.max(dim=1).values
        return min_per_day, max_per_day
        raise NotImplementedError

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        day_averages = self.data.mean(dim=1)
        day_to_day_differences = day_averages[1:] - day_averages[:-1]
        largest_drop = day_to_day_differences.min()
        return largest_drop # Return as a tensor of size (1,
        raise NotImplementedError

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        day_averages = self.data.mean(dim=1)
        differences = torch.abs(self.data - day_averages.unsqueeze(1))
        most_extreme_day_idx = differences.max(dim=1).indices
        most_extreme_day = self.data[torch.arange(self.data.size(0)), most_extreme_day_idx]
        return most_extreme_day  # Return as a tensor of size (num_days,)
        raise NotImplementedError

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        if k <= 0 or k > self.data.size(0):
            raise ValueError("k must be a positive integer and less than or equal to the number of days in the dataset.")
        last_k_days = self.data[-k:]
        max_per_day = last_k_days.max(dim=1).values
        return max_per_day  # Return as a tensor of size (k,)
        raise NotImplementedError

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        if k <= 0 or k > self.data.size(0):
            raise ValueError("k must be a positive integer and less than or equal to the number of days in the dataset.")
        last_k_days = self.data[-k:]
        predicted_temperature = last_k_days.mean()
        return predicted_temperature  # Return as a tensor of size (1,)
        raise NotImplementedError

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
         # Calculate sum of absolute differences for each day
        differences = torch.abs(self.data - t).sum(dim=1)
    
        # Find the day with minimum difference
        closest_day_index = differences.argmin()
        return closest_day_index
        raise NotImplementedError
