library(shiny)
library(plotly)

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Mietview"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      selectizeInput("xVarSel", 'X-Axis', choices=names(data), selected="mieteqm", multiple=FALSE),
      selectizeInput("yVarSel", 'Y-Axis', choices=names(data), selected="bjahr", multiple=FALSE),
      hr(),
      selectizeInput("filterVarSel", 'Filter Variable', choices=names(data), selected="bad", multiple=FALSE),
      sliderInput("filterRangeSel", "Range", min = 1,  max = 10, value = c(1, 5)),
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotlyOutput("distPlot")
    )
  )
)
