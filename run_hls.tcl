#####################################################
# Define constants
set project_name "lstmkws"
set top_level_name "lstm_top"
#####################################################

open_project -reset $project_name

# Add design files
add_files main.cpp
# Add header files
add_files dense.h
add_files lstm1.h
add_files lstm2.h

# Add test bench & files
add_files -tb tb.cpp

# Set the top-level function
set_top $top_level_name

# ########################################################
# Create a solution

open_solution -reset solution1 -flow_target vitis
# Define technology and clock rate
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period "200MHz"

# Run C simulation
csim_design

# Run synthesis
csynth_design
  
# Export RTL to IP catalog
export_design -format ip_catalog

# Open the synthesis report file
set report_file "$project_name/solution1/syn/report/${top_level_name}_csynth.rpt"
if {[file exists $report_file]} {
    set fp [open $report_file r]
    set file_content [read $fp]
    close $fp
    puts "Synthesis Report:\n$file_content"
} else {
    puts "Error: Synthesis report file not found at $report_file"
}

# Close the Vitis shell
exit
