
import pandas as pd
import geopandas as gpd
from bokeh.io import output_notebook, show, push_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, GeoJSONDataSource, LinearColorMapper, ColorBar, FixedTicker, Div
from bokeh.plotting import figure
from bokeh.palettes import Blues9 as palette
from bokeh.transform import transform
from bokeh.themes import Theme
from ipywidgets import interact
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('wfpvam_foodprices.xlsx', engine='openpyxl')

# Create a ColumnDataSource for the line plot
source = ColumnDataSource(data=dict(x=[], y=[]))

# Create a figure for the line plot without a title
p_line = figure(x_axis_type="datetime", width=800, height=400)
p_line.line('x', 'y', source=source)

# Create a Div widget for the title
title_div = Div(text="")

# Load the shapefile
afg_map = gpd.read_file('gadm36_AFG_1.shp')

# Filter the data for Afghanistan and "Wheat - Retail"
afg_data = data[data['Country'] == 'Afghanistan']
afg_data_recent = afg_data[afg_data['mp_year'] == afg_data['mp_year'].max()]
afg_data_recent = afg_data_recent[afg_data_recent['cm_name'] == 'Wheat - Retail']

# Calculate the average price for each region
afg_data_recent = afg_data_recent.groupby('Region')['mp_price'].mean().reset_index()

# Merge the average price data with the shapefile
afg_map = afg_map.merge(afg_data_recent, left_on='NAME_1', right_on='Region', how='left')

# Convert the GeoDataFrame to a GeoJSONDataSource
geosource = GeoJSONDataSource(geojson=afg_map.to_json())

# Create a color mapper
mapper = LinearColorMapper(palette=palette, low=afg_map['mp_price'].min(), high=afg_map['mp_price'].max())

# Create a figure for the map
p_map = figure(title = "Recent Change in Price of Wheat - Retail in Afghanistan Provinces\nTime Period: {} - {}".format(afg_data['mp_year'].min(), afg_data['mp_year'].max()), 
           tools="wheel_zoom,box_zoom,reset", 
           x_axis_location=None, y_axis_location=None,
           width=800, height=400)

# Remove grid lines
p_map.grid.grid_line_color = None

# Add the map to the figure
p_map.patches('xs','ys', source=geosource,
          fill_color = transform('mp_price', mapper),
          line_color = "black", 
          line_width = 0.25, 
          fill_alpha = 1,
          hover_fill_color="firebrick")

# Add hover tool
hover = HoverTool(tooltips=[
    ("Province", "@NAME_1"),
    ("Price", "@mp_price")
])
p_map.add_tools(hover)

# Define the price ranges for the colors
price_ranges = np.linspace(afg_map['mp_price'].min(), afg_map['mp_price'].max(), len(palette)+1)

# Add color bar
color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                     ticker=FixedTicker(ticks=price_ranges))
p_map.add_layout(color_bar, 'right')

# Display the plots in the notebook
output_notebook()
show(column(title_div, p_line, p_map), notebook_handle=True)

# Update function for the line plot
def update(country, commodity):
    subset = data[(data['Country'] == country) & (data['cm_name'] == commodity)]
    subset['Date'] = pd.to_datetime(subset['mp_year'].astype(str) + '-' + subset['mp_month'].astype(str))
    subset = subset.sort_values('Date')  # Sort the data by date
    avg_price = subset.groupby('Date')['mp_price'].mean()  # Calculate the average price for each month
    source.data = dict(
        x=avg_price.index,
        y=avg_price.values
    )
    title_div.text = f"Average Price Trend for {commodity} in {country}"
    push_notebook()

# Create dropdown menus for selecting the country and the commodity
interact(update, country=data['Country'].unique().tolist(), commodity=data['cm_name'].unique().tolist())

# Filter the data for Wheat - Retail
wheat_data = data[data['cm_name'] == 'Wheat - Retail']

# Create separate dataframes for Afghanistan
afg_data = wheat_data[wheat_data['Country'] == 'Afghanistan']

# Convert the year and month to a datetime
afg_data['date'] = pd.to_datetime(afg_data['mp_year'].astype(str) + '-' + afg_data['mp_month'].astype(str) + '-01')

# Group by date and calculate the mean price
afg_data_grouped = afg_data.groupby('date')['mp_price'].mean().reset_index()

# Set the date as the index
afg_data_grouped.set_index('date', inplace=True)

# Decompose the time series
result = seasonal_decompose(afg_data_grouped['mp_price'], model='multiplicative', period=12)

# Increase the size of the plot
plt.rcParams['figure.figsize'] = [10, 8]

# Plot the decomposed time series
result.plot()
plt.show()
 
