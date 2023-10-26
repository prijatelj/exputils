## Visuals

Code for visualizing data.

### Features

#### TODO

- Plotly version of that seaborn/matplotlib code for 2d scatter plot, KDE, and histogram analysis (pairwise).
    - 3d version of the above that all roates together
- plotly PCA visualization for convenience
- plotly time series visuals
    - e.g. that transcription visual sam had.
- Accessibility for best visibility
    - Have an extensive marker set
    - write the code that makes equally spaced colors using HSV and converting to RGB, Hex codes, etc.
        - `numpy.linspace(min_color_value, max_color_value, number_of_colors)` posisbly with a + 2 to number of colors and taking all the values except for the first and last.
        - maximize difference in visual space that does not rely on color (markers, line style)
            - aquire a set of distinct point symbol that indicate a center
            - if the center doesn't matter, then could use characters.
        - Then do gray scale, probably no more than 4 values, to keep them distinct from one another.
            Black to light gray, cuz we cannot do white, as that is the background color.
        - Then keep it safe by including only colors that are most visible to all color vision.
        - And then as necessary, add more Hues.
            - At this point, accessiblity starts to decrease substantially.
- Plotly code for that character sequence probability per pixel (group).
