library(shiny)
library(readxl)
library(plotly)
library(DT)


# Load data
#data <- read_excel("../data/mietspiegel_muc.xlsx", sheet="Mietspiegel")
data <- read_excel("../data/mietspiegel_muc_2003.xlsx", sheet="Mietspiegel")


# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Münchner Mietspiegel"),
  
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
      plot_ly(x=~get(input$xVarSel), y=~get(input$yVarSel), color = ~get(input$cVarSel),
              colors="Set2", type="scatter", mode="markers", opacity=1.0, customdata = seq(1, nrow(fdata())), source="scatter") %>%
      layout(title = sprintf("Scatter Plot"), xaxis = list(title = input$xVarSel), yaxis = list(title = input$yVarSel)) %>%
      layout(showlegend=TRUE, legend=list(title=list(text=sprintf('<b>%s</b>', input$cVarSel), orientation = 'v'))) %>%
      event_register("plotly_selecting")
  })
  
  output$nPoints <- renderText({
    nrow(fdata())
  })
  
  # Handle plotly selections
  output$table <- DT::renderDataTable(server=FALSE, {
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

# Run the application 
shinyApp(ui = ui, server = server)
