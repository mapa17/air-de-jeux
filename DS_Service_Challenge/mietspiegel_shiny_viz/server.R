library(shiny)
library(ggplot2)
library(plotly)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  
  output$distPlot <- renderPlotly({
    
    data %>%
      select(X=input$xVarSel, Y=input$yVarSel, C=input$filterVarSel) %>%
      filter( C >= input$filterRangeSel[1] & C <= input$filterRangeSel[2]) %>%
      plot_ly(x = ~X, y=~Y, type="scatter", mode="markers") %>%
      layout(title = sprintf("Scatter Plot"), xaxis = list(title = input$xVarSel), yaxis = list(title = input$yVarSel))
  })
  
  observeEvent(input$filterVarSel, {
    mi <- min(data[, input$filterVarSel])
    ma <- max(data[, input$filterVarSel])
    updateSliderInput(session, "filterRangeSel", min=mi, max=ma, value = c(mi, ma) )
  })
}