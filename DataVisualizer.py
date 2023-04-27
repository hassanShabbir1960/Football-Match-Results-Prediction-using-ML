#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DataVisualizer:
    
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    
    def plot_home_away_wins(self):
        """
        Plots a grouped bar chart showing the number of wins in home and away matches per team
        
        Returns:
        None
        """
          # Group the data by team and venue, and count the number of wins
        team_wins = self.df[self.df['Result'] == 'W'].groupby(['Team', 'Venue'])['Result'].count().reset_index(name='Count')

        # Pivot the data to get separate columns for home and away wins
        team_wins = team_wins.pivot(index='Team', columns='Venue', values='Count').reset_index()

        # Create a grouped bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=team_wins['Team'],
            y=team_wins['Away'],
            name='Away Wins',
            marker_color='indianred'
        ))

        fig.add_trace(go.Bar(
            x=team_wins['Team'],
            y=team_wins['Home'],
            name='Home Wins',
            marker_color='lightsalmon'
        ))

        fig.update_layout(barmode='group', title='Number of Wins in Home and Away Matches per Team', xaxis_title="Team", yaxis_title="Number of Wins")

        fig.show()
        
    def plot_goals_scored_heatmap(self):
        """
        Create a heatmap plot using Plotly to display the goals scored by each team over time.

        Parameters:
        None.

        Returns:
        None. The function displays the plot using the `show()` method of the Plotly `Figure` object.
        """
        pivot_table = pd.pivot_table(self.df, index='Team', columns='Season', values='GF', aggfunc='sum')
        data = go.Heatmap(z=pivot_table.values,
                          x=pivot_table.columns,
                          y=pivot_table.index,
                          colorscale='YlGnBu')

        layout = go.Layout(title='Goals Scored by Team Over Time',
                           xaxis=dict(title='Season'),
                           yaxis=dict(title='Team'))

        fig = go.Figure(data=[data], layout=layout)
        fig.show()
    
    def plot_expected_goals_vs_actual_goals(self):
        """
        Create a scatter plot using Plotly to display the relationship between expected goals and actual goals
        for each team in the English Premier League.

        Parameters:
        None.

        Returns:
        None. The function displays the plot using the `show()` method of the Plotly `Figure` object.
        """
        fig = px.scatter(self.df, x='xG', y='GF', color='Team', opacity=0.7,
                          color_discrete_sequence=px.colors.qualitative.Dark24,
                         hover_data=['Team', 'Opponent', 'Round', 'Season'])

        fig.update_layout(title='Expected Goals vs Actual Goals',
                          xaxis_title='Expected Goals',
                          yaxis_title='Actual Goals')

        fig.show()
        
    
    def plot_team_results(self):
        """
        Create a grouped bar chart using Plotly to display the number of wins, losses, and draws for all teams.

        Parameters:
        None.

        Returns:
        None. The function displays the plot using the `show()` method of the Plotly `Figure` object.
        """
        team_results = self.df.groupby(['Team', 'Result']).size().reset_index(name='Count')

        fig = px.bar(team_results, x='Team', y='Count', color='Result', barmode='group',
                     title='Number of Wins, Losses, and Draws for All Teams')

        fig.show()
    
    def plot_shots_vs_goals(self):
        """
        Create a stacked bar chart using Plotly to display the number of shots and goals scored by each team.

        Parameters:
        None.

        Returns:
        None. The function displays the plot using the `show()` method of the Plotly `Figure` object.
        """
        team_shots_goals = self.df.groupby('Team').agg({'Sh': 'sum', 'GF': 'sum'}).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=team_shots_goals['Team'],
            y=team_shots_goals['Sh'],
            name='Shots'
        ))

        fig.add_trace(go.Bar(
            x=team_shots_goals['Team'],
            y=team_shots_goals['GF'],
            name='Goals Scored'
        ))

        fig.update_layout(barmode='stack', title='Shots vs Goals Scored by Team',
                          xaxis_title='Team', yaxis_title='Number')
        fig.show()
        
    def plot_avg_possession(self):
        """
        Create a bar chart using Plotly to display the average possession by team.

        Parameters:
        None.

        Returns:
        None. The function displays the plot using the `show()` method of the Plotly `Figure` object.
        """
        avg_possession = self.df.groupby('Team')['Poss'].mean().reset_index()
        avg_possession = avg_possession.sort_values('Poss', ascending=False)

        fig = px.bar(avg_possession, x='Team', y='Poss', title='Average Possession per Team',
                     color_discrete_sequence=['#2a9d8f'])

        fig.show()
        
        
    def plot_top_goal_scorers(self):
        """
        Create a horizontal bar chart using Plotly to display the top 5 goal scorers.

        Parameters:
        None.

        Returns:
        None. The function displays the plot using the `show()` method of the Plotly `Figure` object.
        """
        goals_scored = self.df.groupby('Captain')['GF'].sum().reset_index()
        goals_scored = goals_scored.sort_values('GF', ascending=False)
        top_5 = goals_scored.iloc[:5]

        fig = px.bar(top_5, x='GF', y='Captain', orientation='h',
                     title='Number of Goals Scored by Top 5 Players')

        fig.show()
        
    def plot_top_pass_makers(self):
        """
        Create a horizontal bar chart using Plotly to display the top 5 pass makers.

        Parameters:
        None.

        Returns:
        None. The function displays the plot using the `show()` method of the Plotly `Figure` object.
        """
        passes_made = self.df.groupby('Captain')['Poss'].sum().reset_index()
        passes_made = passes_made.sort_values('Poss', ascending=False)
        top_5 = passes_made.iloc[:5]

        fig = px.bar(top_5, x='Poss', y='Captain', orientation='h',
                     title='Number of Passes Made by Top 5 Players')

        fig.show()
        


# In[22]:


if __name__ == "__main__":
    
    dv = DataVisualizer('matches.csv')
    dv.plot_home_away_wins()
    dv.plot_goals_scored_heatmap()
    dv.plot_expected_goals_vs_actual_goals()
    dv.plot_team_results()
    dv.plot_shots_vs_goals()
    dv.plot_avg_possession()
    dv.plot_top_goal_scorers()
    dv.plot_top_pass_makers()


# In[ ]:




