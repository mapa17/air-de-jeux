library(shiny)
library(readxl)

data <- read_excel("./data/mietspiegel_muc.xlsx", sheet="Mietspiegel")

# Run the application 
shinyApp(ui = ui, server = server)
