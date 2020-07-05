library(shiny)
library(ggplot2)
library(plotly)
library(DT)

# Define server logic required to draw a histogram
server <- function(input, output, session) {

  # Filter data
  fdata <- reactive({
    return(data %>% filter( get(input$filterVarSel) >= input$filterRangeSel[1] & get(input$filterVarSel) <= input$filterRangeSel[2]))
  })
  
  # Generate Plot
  output$distPlot <- renderPlotly({
    fdata() %>%
      mutate(!!input$cVarSel := as.factor(get(input$cVarSel))) %>%
      plot_ly(x=~get(input$xVarSel), y=~get(input$yVarSel), color = ~get(input$cVarSel), colors="Set2", type="scatter", mode="markers", opacity=1.0, customdata = seq(1, nrow(fdata())), source="scatter") %>%
      layout(title = sprintf("Scatter Plot"), xaxis = list(title = input$xVarSel), yaxis = list(title = input$yVarSel)) %>%
      layout(showlegend=TRUE, legend=list(title=list(text=sprintf('<b>%s</b>', input$cVarSel), orientation = 'v'))) %>%
      # NOTE: webGL not working within my browser
      #toWebGL() %>% event_register("plotly_selecting")
      event_register("plotly_selecting")
  })
  
  output$nPoints <- renderText({
    nrow(fdata())
  })
  
  # Handle plotly selections
  output$table <- DT::renderDataTable({
    selection <- event_data("plotly_selecting", source="scatter")
    if( is.null(selection)){
      fdata()
    } else {
      fdata() %>% slice(selection$customdata)
    }
  }, extensions="Buttons", options=list(dom="Bfrstrip", buttons=list("pdf", "csv", "excel")))
  
  # Adjust range filter to selected feature
  observeEvent(input$filterVarSel, {
    mi <- min(data[, input$filterVarSel])
    ma <- max(data[, input$filterVarSel])
    updateSliderInput(session, "filterRangeSel", min=mi, max=ma, value = c(mi, ma) )
  })
}