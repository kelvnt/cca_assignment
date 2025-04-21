import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment

import sys
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


REWARD_POLICY = {
    "choice_1": 3,
    "choice_2": 2,
    "choice_3": 1,
    "none_of_choices": -10  # Penalty to strongly avoid unassigned
}


class AssignmentSolver:
    """
    A solver to assign students to CCAs based on their preferences and
    CCA vacancies, using the Hungarian algorithm with preference-based reassignment.
    """
    def __init__(
        self,
        students_file: str,
        vacancies_file: str,
        reward_policy: Dict=REWARD_POLICY
    ):
        """
        Initializes the AssignmentSolver with paths to CSV files.

        Args:
            students_file (str): File path to the student CSV data.
            vacancies_file (str): File path to the vacancies CSV data.
        """
        self.reward_policy = reward_policy
        self.students_file = students_file
        self.vacancies_file = vacancies_file

        # DataFrames for students and vacancies once loaded
        self.students_df: pd.DataFrame = pd.DataFrame()
        self.vacancies_df: pd.DataFrame = pd.DataFrame()
        self.remaining_students_df: pd.DataFrame = pd.DataFrame()
        self.seats_df: pd.DataFrame = pd.DataFrame()
        self.direct_assigned: List[Tuple[str, str]] = []  # List of (student_id, assigned_cca)

    def load_data(self) -> None:
        """
        Loads and prepares the students and vacancies DataFrames, including date conversion
        and de-duplication of students, then logs the results.
        """
        vacancies_df = pd.read_csv(self.vacancies_file)
        students_df = pd.read_csv(self.students_file)
        students_df["date"] = pd.to_datetime(students_df["date"], dayfirst=True)

        original_N = len(students_df)

        # Sort by date ascending, then keep only the last row per student
        students_df = students_df.sort_values(by="date", ascending=True)
        students_df = students_df.groupby("student_id").last().reset_index()

        logger.info(f"Number of students in the file: {original_N}")
        logger.info(f"Number of duplicate entries: {original_N - len(students_df)}")
        logger.info(f"Number of students remaining: {len(students_df)}\n")

        self.students_df = students_df
        self.vacancies_df = vacancies_df


    def create_reward_matrix_expanded(self, students_df: pd.DataFrame, seats_df: pd.DataFrame) -> np.ndarray:
        """
        Creates a reward matrix for (number_of_students) x (number_of_seats).
        If a seat (i.e., CCA) is in the student's top 3, assign the corresponding reward.
        Otherwise, assign 'none_of_choices' penalty.

        Args:
            students_df (pd.DataFrame): DataFrame containing each student's top 3 choices.
            seats_df (pd.DataFrame): Expanded DataFrame where each row corresponds to a single seat.

        Returns:
            np.ndarray: A 2D numpy array of shape (num_students, num_seats).
        """
        num_students = len(students_df)
        num_seats = len(seats_df)
        reward_matrix = np.full((num_students, num_seats), self.reward_policy.get("none_of_choices", 0.0))

        for i, student in students_df.iterrows():
            for choice in ["choice_1", "choice_2", "choice_3"]:
                cca = student[choice]
                matching_seats = seats_df.index[seats_df["cca"] == cca].tolist()
                for seat_idx in matching_seats:
                    reward_matrix[i, seat_idx] = self.reward_policy.get(choice, 0)
        return reward_matrix


    def expand_vacancies(self, vacancies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expands each CCA into several "seat" rows, according to the number of vacant spots.
        Adds dummy "unassigned" seats if there are fewer seats than students.

        Args:
            vacancies_df (pd.DataFrame): DataFrame with columns ["cca", "filled_spots", "vacant_spots", ...]

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a single seat for a CCA.
        """
        seat_data = []
        for _, row in vacancies_df.iterrows():
            for _ in range(int(row["vacant_spots"])):
                seat_data.append({"cca": row["cca"]})
        # Add dummy seats if necessary
        num_students = len(self.remaining_students_df)
        num_seats = len(seat_data)
        if num_seats < num_students:
            logger.info(f"Adding {num_students - num_seats} dummy seats")
            for _ in range(num_students - num_seats):
                seat_data.append({"cca": "unassigned"})
        seats_df = pd.DataFrame(seat_data)
        logger.info(f"Total CCA vacancies: {len(seats_df)} (including {len(seats_df[seats_df['cca'] == 'unassigned'])} unassigned)")
        return seats_df


    def assign_students(
        self,
        students_df: pd.DataFrame,
        seats_df: pd.DataFrame,
        vacancies_df: pd.DataFrame
    ) -> List[Tuple[str, str, str]]:
        """
        Uses the Hungarian algorithm to assign students to seats, with reassignment to
        prefer choices over 'unassigned' based on remaining vacancies.

        Args:
            students_df (pd.DataFrame): DataFrame containing each student's top 3 choices.
            seats_df (pd.DataFrame): Expanded DataFrame where each row corresponds to a single seat.
            vacancies_df (pd.DataFrame): DataFrame tracking remaining vacancies per CCA.

        Returns:
            List[Tuple[str, str, str]]: A list of (student_id, assigned_cca, remark) for all students.
        """
        reward_matrix = self.create_reward_matrix_expanded(students_df, seats_df)
        cost_matrix = -reward_matrix

        idx_to_student_id = students_df["student_id"].reset_index(drop=True)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Track assignments and vacancies
        assignments = []
        vacancies_copy = vacancies_df.copy()
        unassigned_students = []

        for s_idx, seat_idx in zip(row_ind, col_ind):
            student_id = idx_to_student_id[s_idx]
            cca = seats_df.iloc[seat_idx]["cca"]
            student = students_df[students_df["student_id"] == student_id].iloc[0]
            if cca == "unassigned":
                unassigned_students.append(student_id)
            else:
                # Determine remark based on choice
                if cca == student["choice_1"]:
                    remark = "Assigned first choice"
                elif cca == student["choice_2"]:
                    remark = "Assigned second choice"
                elif cca == student["choice_3"]:
                    remark = "Assigned third choice"
                else:
                    remark = "Randomly assigned none of choices"
                assignments.append((student_id, cca, remark))
                # Decrement vacancy
                if cca in vacancies_copy["cca"].values:
                    cca_index = vacancies_copy[vacancies_copy["cca"] == cca].index[0]
                    vacancies_copy.at[cca_index, "vacant_spots"] = max(
                        vacancies_copy.at[cca_index, "vacant_spots"] - 1, 0
                    )

        logger.info(f"Students assigned to 'unassigned' seats: {len(unassigned_students)}")

        # Reassign unassigned students to preferred CCAs with vacancies
        for student_id in unassigned_students:
            student = students_df[students_df["student_id"] == student_id].iloc[0]
            assigned = False
            # Try preferences in order
            for choice, remark in [
                ("choice_1", "Assigned first choice"),
                ("choice_2", "Assigned second choice"),
                ("choice_3", "Assigned third choice")
            ]:
                cca = student[choice]
                if cca in vacancies_copy["cca"].values:
                    cca_index = vacancies_copy[vacancies_copy["cca"] == cca].index[0]
                    if vacancies_copy.at[cca_index, "vacant_spots"] > 0:
                        assignments.append((student_id, cca, remark))
                        vacancies_copy.at[cca_index, "vacant_spots"] -= 1
                        assigned = True
                        break
            if not assigned:
                # Try any remaining CCA (random assignment)
                available_ccas = vacancies_copy[vacancies_copy["vacant_spots"] > 0]["cca"].tolist()
                if available_ccas:
                    cca = random.choice(available_ccas)
                    cca_index = vacancies_copy[vacancies_copy["cca"] == cca].index[0]
                    assignments.append((student_id, cca, "Randomly assigned none of choices"))
                    vacancies_copy.at[cca_index, "vacant_spots"] -= 1
                else:
                    # No vacancies left
                    assignments.append((student_id, "unassigned", "Unassigned due to lack of vacancies"))

        # Log the cost for debugging
        total_cost = cost_matrix[row_ind, col_ind].sum()
        logger.debug(f"Hungarian algorithm cost: {total_cost}")

        return assignments


    def calculate_assignment_stats(
        self,
        assignments: List[Tuple[str, str, str]],
        students_df: pd.DataFrame,
    ) -> Tuple[int, int, int, int]:
        """
        Counts how many students got their first, second, third, or none of their choices.

        Args:
            assignments (List[Tuple[str, str, str]]): List of (student_id, assigned_cca, remark).
            students_df (pd.DataFrame): DataFrame of student info.

        Returns:
            A 4-tuple of (count_first, count_second, count_third, count_none).
        """
        first_choice_count = 0
        second_choice_count = 0
        third_choice_count = 0
        none_of_choices_count = 0

        for student_id, assigned_cca, _ in assignments:
            row = students_df.loc[students_df["student_id"] == student_id].iloc[0]
            if assigned_cca == row["choice_1"]:
                first_choice_count += 1
            elif assigned_cca == row["choice_2"]:
                second_choice_count += 1
            elif assigned_cca == row["choice_3"]:
                third_choice_count += 1
            else:
                none_of_choices_count += 1

        return first_choice_count, second_choice_count, third_choice_count, none_of_choices_count


    def calculate_reward(
        self,
        assignments: List[Tuple[str, str, str]],
        students_df: pd.DataFrame
    ) -> int:
        """
        Calculates the total reward from assignments, including penalties for none_of_choices assignments.

        Args:
            assignments (List[Tuple[str, str, str]]): List of (student_id, assigned_cca, remark).
            students_df (pd.DataFrame): DataFrame of student info.

        Returns:
            int: The total reward achieved.
        """
        total_reward = 0
        for student_id, cca, _ in assignments:
            student = students_df.loc[students_df["student_id"] == student_id].iloc[0]
            for choice in ["choice_1", "choice_2", "choice_3"]:
                if student[choice] == cca:
                    total_reward += self.reward_policy.get(choice, 0)
                    break
            else:
                total_reward += self.reward_policy.get("none_of_choices", 0)
        return total_reward


    def run(self):
        """
        Executes the solver pipeline:
          1) Loads data.
          2) Handles direct assignments and vacancy adjustments.
          3) Expands seats.
          4) Assigns students using the Hungarian algorithm with reassignment.
          5) Combines direct assignments with optimized results.
          6) Calculates and logs stats.
          7) Outputs final data to CSV with remarks.

        Returns:
            pandas dataframe of final assignments
        """
        self.load_data()

        for i, row in self.students_df.iterrows():
            direct_cca = row.get("direct_assignment", "")
            if isinstance(direct_cca, str) and direct_cca.strip():
                self.direct_assigned.append((row["student_id"], direct_cca, "Directly assigned"))
                if direct_cca in self.vacancies_df["cca"].values:
                    cca_index = self.vacancies_df[self.vacancies_df["cca"] == direct_cca].index[0]
                    old_vacancy = self.vacancies_df.at[cca_index, "vacant_spots"]
                    self.vacancies_df.at[cca_index, "vacant_spots"] = max(old_vacancy - 1, 0)

        direct_assigned_ids = [item[0] for item in self.direct_assigned]
        self.remaining_students_df = self.students_df[~self.students_df.student_id.isin(direct_assigned_ids)]
        self.remaining_students_df = self.remaining_students_df.reset_index(drop=True)

        logger.info(f"Number of direct assignment students: {len(self.direct_assigned)}")
        logger.info(f"Number of students to assign: {len(self.remaining_students_df)}")

        self.seats_df = self.expand_vacancies(self.vacancies_df)

        assignments = self.assign_students(
            self.remaining_students_df,
            self.seats_df,
            self.vacancies_df
        )

        final_assignments = self.direct_assigned + assignments

        total_reward = self.calculate_reward(final_assignments, self.students_df)
        first_c, second_c, third_c, none_c = self.calculate_assignment_stats(final_assignments, self.students_df)

        logger.info(f"\nTotal Students: {len(self.students_df)}")
        logger.info(f"Total Reward: {total_reward}\n")
        logger.info(f"Direct assignments: {len(self.direct_assigned)}")
        logger.info(f"Assigned to 1st choice: {first_c}")
        logger.info(f"Assigned to 2nd choice: {second_c}")
        logger.info(f"Assigned to 3rd choice: {third_c}")
        logger.info(f"Assigned to none of their choices: {none_c}\n")

        # Prepare CSV with remarks
        assignment_df = pd.DataFrame(final_assignments, columns=["student_id", "assigned_cca", "remarks"])
        out = pd.merge(
            self.students_df,
            assignment_df,
            on="student_id",
            how="left"
        )

        return out


def main():
    solver = AssignmentSolver(
        students_file="data/student_list.csv",
        vacancies_file="data/cca_vacancies.csv",
    )
    solver.run(
        output_file="data/final_assignments.csv"
    )


if __name__ == "__main__":
    main()