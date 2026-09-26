import numpy as np


class StatisticsCounter:
    __slots__ = (
        "checking_times",
        "ctd",
        "generation_times",
        "mtn_1",
        "mtn_2",
        "rejected_solutions",
        "round",
        "samples",
        "steps",
        "total_times",
    )

    def __init__(self, rounds):
        self.steps = np.zeros(rounds)
        self.rejected_solutions = np.zeros(rounds)
        self.samples = np.ndarray(rounds, object)
        self.checking_times = np.ndarray(rounds, object)
        self.generation_times = np.ndarray(rounds, object)
        self.total_times = np.ndarray(rounds, object)
        self.mtn_1 = np.zeros(rounds)
        self.mtn_2 = np.zeros(rounds)
        self.ctd = np.zeros(rounds)
        self.round = 0

    def record_Round(
        self,
        steps,
        rejected_solutions,
        samples,
        checking_times,
        generation_times,
        total_times,
        mtn_1,
        mtn_2,
        ctd,
    ):
        self.steps[self.round] = steps
        self.rejected_solutions[self.round] = rejected_solutions
        self.samples[self.round] = samples
        self.checking_times[self.round] = checking_times
        self.generation_times[self.round] = generation_times
        self.total_times[self.round] = total_times
        self.mtn_1[self.round] = mtn_1
        self.mtn_2[self.round] = mtn_2
        self.ctd[self.round] = ctd
        self.round += 1

    def print_statistics(self, logLevel):
        samples = np.concatenate(self.samples)
        checking_times = np.concatenate(self.checking_times)
        generation_times = np.concatenate(self.generation_times)
        total_times = np.concatenate(self.total_times)
        match logLevel:
            case 0:
                print(
                    f"{np.average(self.steps):.4f}, {np.std(self.steps):.4f}, {
                        np.average(self.rejected_solutions):.4f}, {
                        np.std(self.rejected_solutions):.4f}, {
                        np.average(samples):.4f}, {np.std(samples):.4f}, {
                        np.average(checking_times):.4f}, {np.std(checking_times):.4f}, {
                        np.average(generation_times):.4f}, {
                        np.std(generation_times):.4f}, {np.average(total_times):.4f}, {
                        np.std(total_times):.4f}, {np.average(self.mtn_1):.4f}, {
                        np.average(self.mtn_2):.4f}, {np.average(self.ctd):.4f}"
                )
            case 1:
                print(
                    f"Average Steps per Round:{
                        np.average(self.steps):.4f}, Standard deviation:{
                        np.std(
                            self.steps
                        ):.4f}\nAverage Percent of rejected Policy Fixes:{
                        np.average(self.rejected_solutions):.4f}, Standard deviation:{
                        np.std(self.rejected_solutions):.4f}\nAverage Samples per Step:{
                        np.average(samples):.4f}, Standard deviation:{
                        np.std(samples):.4f}\nAverage time spent checking per step:{
                        np.average(checking_times):.4f}, standard deviation:{
                        np.std(
                            checking_times
                        ):.4f}\nAverage time spent generating a policy fix per step:{
                        np.average(generation_times):.4f}, standard deviation:{
                        np.std(generation_times):.4f}\nAverage time per step:{
                        np.average(total_times):.4f}, Standard deviation:{
                        np.std(total_times):.4f}\nAverage amount of MTN1 per Round:{
                        np.average(self.mtn_1):.4f}\nAverage amount of MTN2 per Round:{
                        np.average(self.mtn_2):.4f}\nAverage amount of CTD per Round:{
                        np.average(self.ctd):.4f}"
                )
