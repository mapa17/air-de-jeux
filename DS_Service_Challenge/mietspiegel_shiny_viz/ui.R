library(shiny)
library(plotly)
library(DT)

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Mietview"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(position = "right",
    sidebarPanel(
      selectizeInput("xVarSel", 'X-Axis', choices=names(data), selected="mieteqm", multiple=FALSE),
      selectizeInput("yVarSel", 'Y-Axis', choices=names(data), selected="bjahr", multiple=FALSE),
      selectizeInput("cVarSel", 'Color', choices=names(data), selected="lage", multiple=FALSE),
      hr(),
      selectizeInput("filterVarSel", 'Filter Variable', choices=names(data), selected="flaeche", multiple=FALSE),
      sliderInput("filterRangeSel", "Range", min = 1,  max = 10, value = c(1, 5)),
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      plotlyOutput("distPlot"),
      paste0('Filtered Points:'),
      textOutput("nPoints"),
      hr(),
      tags$div(tags$strong("Use the selection tools to filter the scatter points viewed in the table.")),
      hr(),
      DT::dataTableOutput("table")
      
    )
  )
)
