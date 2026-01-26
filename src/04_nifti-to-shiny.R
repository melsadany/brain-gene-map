# app.R
library(shiny)
library(tidyverse)
library(RNifti)
device <- ifelse(grepl("/LSS/", system("cd &pwd", intern = T)), "IDAS", "argon")
source(paste0(ifelse(device == "IDAS", "~/LSS", "/Dedicated"),"/jmichaelson-wdata/msmuhammad/msmuhammad-source.R"))

# my custom color palette
colors <- list(
  primary = "#4782b4",
  accent = "#ff4600",
  success = "#39C08F",
  secondary = "#627899",
  warm = "#C1624A",
  light_blue = "#88ADE1",
  light_warm = "#F3B199",
  dark = "#55433C",
  muted = "#A6665F",
  cyan = "#00C0C5"
)

# Red-blue color palette for intensity
create_rb_palette <- function(n = 256) {
  colorRampPalette(c("#4782b4", "white", "#ff4600"))(n)
}

# ===== UI =====
ui <- fluidPage(
  tags$head(
    tags$style(HTML(paste0("
      /* Design System - Custom Colors */
      :root {
        --primary: ", colors$primary, ";
        --accent: ", colors$accent, ";
        --success: ", colors$success, ";
        --secondary: ", colors$secondary, ";
        --warm: ", colors$warm, ";
        --light-blue: ", colors$light_blue, ";
        --light-warm: ", colors$light_warm, ";
        --dark: ", colors$dark, ";
        --bg-light: #f5f5f5;
        --text-dark: #1a1a1a;
        --text-light: #666;
        --border: #ddd;
      }
      
      * {
        box-sizing: border-box;
      }
      
      body {
        font-family: 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f5f5f5 0%, #ececec 100%);
        color: var(--text-dark);
        margin: 0;
        padding: 0;
      }
      
      .navbar {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%) !important;
        padding: 1.25rem 2rem !important;
        box-shadow: 0 2px 8px rgba(71, 130, 180, 0.2);
      }
      
      .navbar-brand {
        font-weight: 700;
        font-size: 1.5rem;
        letter-spacing: -0.5px;
        color: white !important;
      }
      
      .container-main {
        max-width: 1600px;
        margin: 2rem auto;
        padding: 0 1rem;
      }
      
      /* Sidebar */
      .sidebar {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border-top: 4px solid var(--accent);
      }
      
      .sidebar h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 1rem;
        margin-top: 0;
      }
      
      .form-group label {
        font-weight: 500;
        color: var(--text-dark);
        font-size: 0.95rem;
      }
      
      select, input {
        border: 2px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 0.75rem !important;
        font-size: 0.95rem;
        transition: all 0.2s ease;
      }
      
      select:focus, input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(255, 70, 0, 0.15) !important;
        outline: none;
      }
      
      /* Main content area - HORIZONTAL LAYOUT */
      .content-area {
        display: flex;
        flex-direction: column;
        gap: 2rem;
      }
      
      .main-grid {
        display: grid;
        grid-template-columns: 3fr 1fr;  /* CHANGED from 2fr 1fr to 3fr 1fr */
        gap: 2rem;
        margin-bottom: 2rem;
      }
      
      .panel {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        overflow: hidden;
      }
      
      .panel-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
      }
      
      .panel-body {
        padding: 1.5rem;
      }
      
      /* Crosshair controls */
      .crosshair-controls {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
      }
      
      .coord-input {
        display: flex;
        flex-direction: column;
        align-items: center;
      }
      
      .coord-input label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
        text-align: center;
        width: 100%;
      }
      
      .coord-input input {
        padding: 0.6rem !important;
        font-size: 0.9rem;
        text-align: center;
        font-family: 'Monaco', 'Courier New', monospace;
        max-width: 100px;
        margin: 0 auto;
      }
      
      /* Three-column slice views */
      .slices-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-bottom: 1.5rem;
      }
      
      .slice-view {
        border: 2px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
        background: #f0f0f0;
      }
      
      .slice-header {
        background: var(--light-blue);
        color: white;
        padding: 0.75rem;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
      }
      
      .slice-plot {
        width: 100%;
        height: 350px;
        cursor: crosshair;
      }
      
      /* Intensity and annotations panel */
      .info-panel {
        display: flex;
        flex-direction: column;
        height: 100%;
        min-width: 300px;
      }
      
      .intensity-section {
        background: linear-gradient(135deg, var(--light-blue) 0%, ", colors$primary, " 100%);
                          border-radius: 8px;
                          padding: 1.25rem;  /* CHANGED from 1.5rem to 1.25rem */
                            color: white;
                          box-shadow: 0 4px 12px rgba(71, 130, 180, 0.2);
                          margin-bottom: 1.5rem;
                          }
      
      .intensity-value {
        font-size: 1.6rem;
          font-weight: 700;
        font-family: 'Monaco', 'Courier New', monospace;
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0 1rem 0;
        word-break: break-all;  /* ADD THIS - prevents text overflow */
      }
      
      .color-scale {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.3);
      }
      
      .color-scale-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: rgba(255, 255, 255, 0.9);
      }
      
      .color-bar {
        height: 12px;
        background: linear-gradient(to right, blue, white, red);
        border-radius: 6px;
        width: 100%;
      }
      
      /* Atlas annotations */
        .annotations-section {
          background: white;
          border-radius: 8px;
          padding: 1.25rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          border-left: 4px solid var(--success);
          flex-grow: 1;
        }
      
      /* Atlas table styling */
        .atlas-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.85rem;
        }
      
      .atlas-table th {
        background: var(--success);
        color: white;
        padding: 0.6rem;
          text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
          letter-spacing: 0.5px;
      }
      
      .atlas-table td {
        padding: 0.6rem;
          border-bottom: 1px solid var(--border);
      }
      
      .atlas-table tr:hover {
        background: rgba(57, 192, 143, 0.1);
      }
      
      .atlas-label {
        font-weight: 500;
        color: var(--text-dark);
      }
      
      .status-found {
        color: var(--success);
        font-weight: 600;
      }
      
      .status-not-found {
        color: var(--warm);
        font-style: italic;
      }
      
      /* Coordinate display */
        .coord-display {
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 0.9rem;
          color: rgba(255, 255, 255, 0.9);
          margin-bottom: 1rem;
          background: rgba(0, 0, 0, 0.1);
          padding: 0.5rem 0.75rem;
          border-radius: 4px;
          display: inline-block;
        }
      
      /* Buttons */
        .btn-primary {
          background: var(--accent) !important;
          border: none !important;
          border-radius: 6px !important;
          font-weight: 600;
          padding: 0.75rem 1.5rem !important;
          transition: all 0.2s ease;
          color: white !important;
        }
      
      .btn-primary:hover {
        background: var(--warm) !important;
        box-shadow: 0 4px 12px rgba(255, 70, 0, 0.3);
      }
      
      /* Status message */
        .status-message {
          padding: 0.75rem;
          background: var(--light-blue);
          color: white;
          border-radius: 4px;
          margin-top: 0.75rem;
          font-size: 0.9rem;
          font-weight: 500;
        }
      
      .status-error {
        background: var(--warm);
      }
      ")))
  ),
  
  navbarPage(
    title = "Interactive Brain Atlas Viewer",
    windowTitle = "Brain Atlas Viewer for ",
    theme = bslib::bs_theme(version = 5, primary = colors$primary, secondary = colors$secondary),
    
    tabPanel(
      "Viewer",
      class = "container-main",
      
      # Control panel
      div(
        class = "sidebar",
        h3("Select Gene & Coordinates"),
        fluidRow(
          column(8,
                 selectInput("nifti_select",
                             label = "Choose Gene Expression Map",
                             choices = NULL,
                             width = "100%"
                 )
          ),
          column(4,
                 actionButton("load_btn",
                              "Load Map",
                              class = "btn btn-primary",
                              width = "100%",
                              style = "margin-top: 25px;"
                 )
          )
        ),
        div(
          textOutput("file_status_msg"),
          style = "color: var(--text-light); font-size: 0.9rem; margin-top: 0.5rem;"
        ),
        div(  # ADD THIS EXPLANATORY BOX
          style = "margin-top: 1rem; padding: 0.75rem; background: rgba(71, 130, 180, 0.1); border-radius: 6px; border-left: 3px solid var(--accent);",
          p(
            strong("Interpretation Guide:", style = "color: var(--accent);"),
            br(),
            span("• ", style = "color: var(--accent); font-weight: bold;"),
            span("Positive values: ", style = "font-weight: 500;"),
            "Predicted high gene expression pattern",
            br(),
            span("• ", style = "color: var(--accent); font-weight: bold;"),
            span("Negative values: ", style = "font-weight: 500;"),
            "Predicted low gene expression pattern",
            style = "margin: 0; font-size: 0.8rem; line-height: 1.4; color: var(--text-dark);"
          )
        )
      ),
      
      # Main viewer & data - SIDE BY SIDE LAYOUT
      div(
        class = "main-grid",
        
        # Left column: Slices (now wider - 75% of space)
        div(
          class = "panel",
          div(class = "panel-header", "Predicted Gene Expression Maps"),
          div(
            class = "panel-body",
            
            # Crosshair position inputs
            div(
              class = "crosshair-controls",
              div(
                class = "coord-input",
                tags$label("X (Leftt-Right)", `for` = "x_coord"),
                numericInput("x_coord", NULL, value = 45, min = 0, max = 90, 
                             width = "100%")  # ADD width parameter
              ),
              div(
                class = "coord-input",
                tags$label("Y (Posterior-Anterior)", `for` = "y_coord"),
                numericInput("y_coord", NULL, value = 54, min = 0, max = 108,
                             width = "100%")  # ADD width parameter
              ),
              div(
                class = "coord-input",
                tags$label("Z (Inferior-Superior)", `for` = "z_coord"),
                numericInput("z_coord", NULL, value = 45, min = 0, max = 90,
                             width = "100%")  # ADD width parameter
              )
            ),
            
            # THREE SLICES SIDE BY SIDE
            div(
              class = "slices-container",
              
              div(
                class = "slice-view",
                div(class = "slice-header", "Axial (XY)"),
                plotOutput("axial_plot", height = "350px", click = "axial_click")
              ),
              
              div(
                class = "slice-view",
                div(class = "slice-header", "Sagittal (YZ)"),
                plotOutput("sagittal_plot", height = "350px", click = "sagittal_click")
              ),
              
              div(
                class = "slice-view",
                div(class = "slice-header", "Coronal (XZ)"),
                plotOutput("coronal_plot", height = "350px", click = "coronal_click")
              )
            )
          )
        ),
        
        # Right column: Intensity & Annotations (now narrower - 25% of space)
        div(
          class = "info-panel",
          
          # Intensity panel with color scale
          div(
            class = "intensity-section",
            h4("Gene Expression", 
               style = "margin-top: 0; margin-bottom: 0.5rem; color: white; font-size: 1rem;"),
            div(
              class = "coord-display",
              textOutput("coord_display", inline = TRUE)
            ),
            div(
              style = "margin: 1rem 0;",
              div(
                style = "font-size: 0.85rem; margin-bottom: 0.25rem;",
                "Gene Expression Score:"
              ),
              div(
                class = "intensity-value",
                textOutput("intensity_value", inline = TRUE)
              )
            ),
            div(  # ADD THIS DIV for the color scale
              class = "color-scale",
              div(
                class = "color-scale-label",
                span("Low"),
                span("High")
              ),
              div(class = "color-bar")
            )
          ),
          
          # Atlas annotations
          div(
            class = "annotations-section",
            div(
              class = "annotations-header",
              "Atlas Annotations"
            ),
            tableOutput("atlas_labels_table")
          )
        )
      )
    )
  )
)

# ===== SERVER =====
server <- function(input, output, session) {
  
  current_nifti <- reactiveVal(NULL)
  atlas_df <- reactiveVal(NULL)
  
  # Create red-blue palette
  rb_palette <- reactive({
    create_rb_palette(256)
  })
  
  # Get list of nifti files at startup
  nifti_dir <- correct_path("/wdata/msmuhammad/projects/brain-gene-map/data/derivatives/gene-nifti-maps")
  nifti_files <- list.files(nifti_dir, pattern = "\\.nii(\\.gz)?$", full.names = TRUE)
  nifti_names <- basename(nifti_files)
  
  # Update dropdown with files
  observe({
    if (length(nifti_files) > 0) {
      display_names <- gsub("\\.nii(\\.gz)?$", "", nifti_names, ignore.case = TRUE)
      choices <- setNames(nifti_files, display_names)
      updateSelectInput(session, "nifti_select", 
                        choices = choices,
                        selected = nifti_files[1])
    } else {
      updateSelectInput(session, "nifti_select", 
                        choices = c("No NIfTI files found" = ""))
    }
  })
  
  output$file_status_msg <- renderText({
    paste("✓ Found", length(nifti_files), "deen expression  map(s)")
  })
  
  # Load atlas data
  atlas_path <- correct_path("/wdata/msmuhammad/refs/labeled-MNI/2mm/voxels-w-labels-R.rds")
  if (file.exists(atlas_path)) {
    atlas_df(readRDS(atlas_path))
  }
  
  # Load NIfTI when selection changes
  observeEvent(input$nifti_select, {
    req(input$nifti_select)
    tryCatch({
      cat("Loading:", input$nifti_select, "\n")
      img <- readNifti(input$nifti_select)
      cat("Image loaded. Dims:", paste(dim(img), collapse=" "), "\n")
      current_nifti(img)
    }, error = function(e) {
      cat("Error:", e$message, "\n")
      showNotification(paste("Error:", e$message), type = "error")
    })
  }, ignoreNULL = FALSE)
  
  # Display coordinate
  output$coord_display <- renderText({
    paste0("(", input$x_coord, ", ", input$y_coord, ", ", input$z_coord, ")")
  })
  
  # Intensity value with color coding
  output$intensity_value <- renderText({
    req(current_nifti())
    img <- current_nifti()
    val <- img[input$x_coord, input$y_coord, input$z_coord]
    if (is.na(val)) "NA" else sprintf("%.6f", val)
  })
  
  # Axial slice (XY plane at Z)
  output$axial_plot <- renderPlot({
    req(current_nifti())
    img <- current_nifti()
    z_idx <- as.integer(input$z_coord)
    
    # Get axial slice (XY plane)
    slice_data <- img[, , z_idx]
    
    # Flip for radiological convention (right on left)
    slice_data <- slice_data[nrow(slice_data):1, ]
    
    # INCREASED MARGINS: bottom, left, top, right
    par(mar = c(4, 4, 3, 1.5), bg = "white")  # CHANGED from c(3.5, 3.5, 2.5, 1)
    
    # Get data range for symmetric color scaling
    data_range <- range(slice_data, na.rm = TRUE)
    max_abs <- max(abs(data_range))
    
    # Plot with slightly smaller font for axis labels
    image(1:nrow(slice_data), 1:ncol(slice_data), slice_data,
          col = rb_palette(),
          xlab = "X (R→L)",
          ylab = "Y (P→A)",
          main = paste("Z =", z_idx),
          asp = ncol(slice_data)/nrow(slice_data),
          zlim = c(-max_abs, max_abs),
          cex.lab = 0.85,  # CHANGED from 0.9 to 0.85
          cex.main = 0.95, # CHANGED from 1 to 0.95
          cex.axis = 0.75) # CHANGED from 0.8 to 0.75
    
    # Add crosshair - account for flipping
    abline(v = nrow(slice_data) - input$x_coord + 1, col = "#4782b4", lwd = 2, lty = 2)
    abline(h = input$y_coord, col = "#ff4600", lwd = 2, lty = 2)
    
    # Add subtle grid
    grid(col = "gray30", lty = 3, lwd = 0.5)
  })
  
  # Sagittal slice (YZ plane at X)
  output$sagittal_plot <- renderPlot({
    req(current_nifti())
    img <- current_nifti()
    x_idx <- as.integer(input$x_coord)
    
    # Get sagittal slice (YZ plane)
    slice_data <- img[x_idx, , ]
    
    # INCREASED MARGINS: bottom, left, top, right
    par(mar = c(4, 4, 3, 1.5), bg = "white")  # CHANGED from c(3.5, 3.5, 2.5, 1)
    
    # Get data range for symmetric color scaling
    data_range <- range(slice_data, na.rm = TRUE)
    max_abs <- max(abs(data_range))
    
    # Plot with slightly smaller font for axis labels
    image(1:nrow(slice_data), 1:ncol(slice_data), slice_data,
          col = rb_palette(),
          xlab = "Y (P→A)",
          ylab = "Z (I→S)",
          main = paste("X =", x_idx),
          asp = ncol(slice_data)/nrow(slice_data),
          zlim = c(-max_abs, max_abs),
          cex.lab = 0.85,  # CHANGED from 0.9 to 0.85
          cex.main = 0.95, # CHANGED from 1 to 0.95
          cex.axis = 0.75) # CHANGED from 0.8 to 0.75
    
    # Add crosshair
    abline(v = input$y_coord, col = "#ff4600", lwd = 2, lty = 2)
    abline(h = input$z_coord, col = "#39C08F", lwd = 2, lty = 2)
    
    # Add subtle grid
    grid(col = "gray30", lty = 3, lwd = 0.5)
  })
  
  # Coronal slice (XZ plane at Y)
  output$coronal_plot <- renderPlot({
    req(current_nifti())
    img <- current_nifti()
    y_idx <- as.integer(input$y_coord)
    
    # Get coronal slice (XZ plane)
    slice_data <- img[, y_idx, ]
    
    # Flip X for radiological convention
    slice_data <- slice_data[nrow(slice_data):1, ]
    
    # INCREASED MARGINS: bottom, left, top, right
    par(mar = c(4, 4, 3, 1.5), bg = "white")  # CHANGED from c(3.5, 3.5, 2.5, 1)
    
    # Get data range for symmetric color scaling
    data_range <- range(slice_data, na.rm = TRUE)
    max_abs <- max(abs(data_range))
    
    # Plot with slightly smaller font for axis labels
    image(1:nrow(slice_data), 1:ncol(slice_data), slice_data,
          col = rb_palette(),
          xlab = "X (L→R)",
          ylab = "Z (I→S)",
          main = paste("Y =", y_idx),
          asp = ncol(slice_data)/nrow(slice_data),
          zlim = c(-max_abs, max_abs),
          cex.lab = 0.85,  # CHANGED from 0.9 to 0.85
          cex.main = 0.95, # CHANGED from 1 to 0.95
          cex.axis = 0.75) # CHANGED from 0.8 to 0.75
    
    # Add crosshair - account for flipping
    abline(v = nrow(slice_data) - input$x_coord + 1, col = "#4782b4", lwd = 2, lty = 2)
    abline(h = input$z_coord, col = "#39C08F", lwd = 2, lty = 2)
    
    # Add subtle grid
    grid(col = "gray30", lty = 3, lwd = 0.5)
  })
  

  # Handle clicks on slices
  observeEvent(input$axial_click, {
    req(current_nifti(), input$axial_click)
    click <- input$axial_click
    dims <- dim(current_nifti())
    
    display_x <- round(click$x)
    display_y <- round(click$y)
    
    # Convert back to actual coordinates (account for flipping)
    actual_x <- dims[1] - display_x + 1
    actual_y <- display_y
    
    if (display_x > 0 && display_x <= dims[1] && display_y > 0 && display_y <= dims[2]) {
      updateNumericInput(session, "x_coord", value = actual_x)
      updateNumericInput(session, "y_coord", value = actual_y)
    }
  })
  
  observeEvent(input$sagittal_click, {
    req(current_nifti(), input$sagittal_click)
    click <- input$sagittal_click
    dims <- dim(current_nifti())
    
    display_y <- round(click$x)
    display_z <- round(click$y)
    
    actual_y <- display_y
    actual_z <- display_z
    
    if (display_y > 0 && display_y <= dims[2] && display_z > 0 && display_z <= dims[3]) {
      updateNumericInput(session, "y_coord", value = actual_y)
      updateNumericInput(session, "z_coord", value = actual_z)
    }
  })
  
  observeEvent(input$coronal_click, {
    req(current_nifti(), input$coronal_click)
    click <- input$coronal_click
    dims <- dim(current_nifti())
    
    display_x <- round(click$x)
    display_z <- round(click$y)
    
    # Convert back to actual coordinates (account for flipping)
    actual_x <- dims[1] - display_x + 1
    actual_z <- display_z
    
    if (display_x > 0 && display_x <= dims[1] && display_z > 0 && display_z <= dims[3]) {
      updateNumericInput(session, "x_coord", value = actual_x)
      updateNumericInput(session, "z_coord", value = actual_z)
    }
  })
  
  # Atlas labels table
  output$atlas_labels_table <- renderTable({
    req(atlas_df())
    
    df <- atlas_df()
    row <- df %>%
      filter(x == input$x_coord, y == input$y_coord, z == input$z_coord)
    
    if (nrow(row) == 0) {
      tibble(
        Atlas = c("CerebrA", "Schaefer", "FS-anat"),
        `Brain Region` = c("Not found", "Not found", "Not found")
      )
    } else {
      labels <- c(
        ifelse(is.na(row$cerebrA_structure), "Not labeled", str_replace_all(as.character(row$cerebrA_structure),"_"," ")),
        ifelse(is.na(row$roi_name), "Not labeled", sub("17Networks ","",str_replace_all(as.character(row$roi_name),"_"," "))),
        ifelse(is.na(row$synthseg_structure), "Not labeled", as.character(row$synthseg_structure))
      )
      
      tibble(
        Atlas = c("CerebrA", "Schaefer", "FS-anat"),
        `Brain Region` = labels
      )
    }
  },
  striped = TRUE,
  hover = TRUE,
  bordered = FALSE,
  spacing = "xs",
  width = "100%",
  align = "ll"
  )
}

shinyApp(ui, server)