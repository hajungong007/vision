ectomodule(PLYWriter_Bin DESTINATION DESTINATION ${PROJECT_NAME}/ecto_cells
    INSTALL     
    # You need a file defining the module (cf. below)
    module.cpp
    # You need a file defining some content for module (cf. below)
    cell.cpp
)
link_ecto(PLYWriter_Bin ${PCL_LIBRARIES})
