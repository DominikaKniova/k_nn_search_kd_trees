#include "visualizer.h"

// inspired by examples from VTK library documentation and examples

vtkSmartPointer<vtkActor> plot_points(std::vector<Point> & data, int dimensionality, int step, double * color){
    auto points = vtkSmartPointer<vtkPoints>::New();

    if (dimensionality == 3){
        for (unsigned int i = 0; i < data.size(); i += step) {
            points->InsertNextPoint(&data[i].p[0]);
        }
    }
    else {
        for (unsigned int i = 0; i < data.size(); i += step) {
            points->InsertNextPoint(data[i].p[0], data[i].p[1], 0.0f);
        }
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    auto vertexGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexGlyphFilter->AddInputData(polydata);
    vertexGlyphFilter->Update();

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color);
    actor->GetProperty()->SetPointSize(5);

    return actor;
}

vtkSmartPointer<vtkActor> plot_range_points(std::vector<Point *> & data, int dimensionality, int step, double *color){
    auto points = vtkSmartPointer<vtkPoints>::New();

    if (dimensionality == 3){
        for (unsigned int i = 0; i < data.size(); i += step) {
            points->InsertNextPoint(&data[i]->p[0]);
        }
    }
    else {
        for (unsigned int i = 0; i < data.size(); i += step) {
            points->InsertNextPoint(data[i]->p[0], data[i]->p[1], 0.0001f);
        }
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    auto vertexGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexGlyphFilter->AddInputData(polydata);
    vertexGlyphFilter->Update();

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color);
    actor->GetProperty()->SetPointSize(6);

    return actor;
}

vtkSmartPointer<vtkActor> plot_range_points(std::deque<Point *> & data, int dimensionality, int step, double *color){
    auto points = vtkSmartPointer<vtkPoints>::New();

    if (dimensionality == 3){
        for (unsigned int i = 0; i < data.size(); i += step) {
            points->InsertNextPoint(&data[i]->p[0]);
        }
    }
    else {
        for (unsigned int i = 0; i < data.size(); i += step) {
            points->InsertNextPoint(data[i]->p[0], data[i]->p[1], 0.0001f);
        }
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    auto vertexGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexGlyphFilter->AddInputData(polydata);
    vertexGlyphFilter->Update();

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color);
    actor->GetProperty()->SetPointSize(6);

    return actor;
}

vtkSmartPointer<vtkActor> plot_query(const VectorXf &query, int dimensionality){
    vtkSmartPointer<vtkNamedColors> colors =
            vtkSmartPointer<vtkNamedColors>::New();

    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();

    if (dimensionality == 3){
        points->InsertNextPoint(query[0], query[1], query[2]);
    }
    else {
        points->InsertNextPoint(query[0], query[1], 0.0001f);
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    auto vertexGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexGlyphFilter->AddInputData(polydata);
    vertexGlyphFilter->Update();

    // Create a mapper and actor
    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(colors->GetColor3d("Cyan").GetData());
    actor->GetProperty()->SetPointSize(10);

    return actor;
}

vtkSmartPointer<vtkActor> plot_range_sph(const VectorXf &query, float rad, int dimensionality){
    vtkSmartPointer<vtkNamedColors> colors =
            vtkSmartPointer<vtkNamedColors>::New();

    // Create a circle
    vtkSmartPointer<vtkRegularPolygonSource> polygonSource =
            vtkSmartPointer<vtkRegularPolygonSource>::New();

    vtkSmartPointer<vtkSphereSource> sphereSource =
            vtkSmartPointer<vtkSphereSource>::New();

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();


    if (dimensionality == 3){
        sphereSource->SetRadius(rad);
        sphereSource->SetCenter(query[0], query[1], query[2]);
        sphereSource->SetPhiResolution(100);
        sphereSource->SetThetaResolution(100);
        mapper->SetInputConnection(sphereSource->GetOutputPort());;
    }
    else {
        polygonSource->SetRadius(rad);
        polygonSource->SetCenter(query[0], query[1], 0.0f);
        polygonSource->SetNumberOfSides(50);
        mapper->SetInputConnection(polygonSource->GetOutputPort());;

    }
    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(colors->GetColor3d("Grey").GetData());
    actor->GetProperty()->SetOpacity(0.8);

    return actor;

}

vtkSmartPointer<vtkActor> plot_visited_leaves(std::vector<std::vector<float>> & leaves, int dimensionality){
    int MIN = 0;
    int MAX = 1;

    vtkSmartPointer<vtkNamedColors> colors =
            vtkSmartPointer<vtkNamedColors>::New();

    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();

    vtkSmartPointer<vtkCellArray> polygons =
            vtkSmartPointer<vtkCellArray>::New();
    if (dimensionality == 2) {
        for (unsigned int i = 0; i < leaves.size(); ++i) {
            // Create the polygon
            vtkSmartPointer<vtkPolygon> polygon =
                    vtkSmartPointer<vtkPolygon>::New();

            points->InsertNextPoint(leaves[i][0 + MIN],  leaves[i][2 + MIN], 0.0);
            points->InsertNextPoint(leaves[i][0 + MAX],  leaves[i][2 + MIN], 0.0);
            points->InsertNextPoint(leaves[i][0 + MAX],  leaves[i][2 + MAX], 0.0);
            points->InsertNextPoint(leaves[i][0 + MIN],  leaves[i][2 + MAX], 0.0);

            polygon->GetPointIds()->SetNumberOfIds(4); //make a quad
            polygon->GetPointIds()->SetId(0, i * 4 + 0);
            polygon->GetPointIds()->SetId(1, i * 4 + 1);
            polygon->GetPointIds()->SetId(2, i * 4 + 2);
            polygon->GetPointIds()->SetId(3, i * 4 + 3);

            // Add the polygon to a list of polygons
            polygons->InsertNextCell(polygon);
        }
    }

    vtkSmartPointer<vtkPolyData> polygonPolyData =
            vtkSmartPointer<vtkPolyData>::New();
    polygonPolyData->SetPoints(points);
    polygonPolyData->SetPolys(polygons);

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polygonPolyData);

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(
            colors->GetColor3d("Green").GetData());
    actor->GetProperty()->SetOpacity(0.3);

    return actor;
}

vtkSmartPointer<vtkActor> plot_range_rect(std::vector<float> & ranges, const VectorXf & query, int dimensionality){
    vtkSmartPointer<vtkNamedColors> colors =
            vtkSmartPointer<vtkNamedColors>::New();

    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();

    vtkSmartPointer<vtkCellArray> polygons =
            vtkSmartPointer<vtkCellArray>::New();
    if (dimensionality == 2) {
        // Create the polygon
        vtkSmartPointer<vtkPolygon> polygon =
                vtkSmartPointer<vtkPolygon>::New();

        //up
        points->InsertNextPoint(query[0] - ranges[0], query[1] + ranges[1], 0.0);
        points->InsertNextPoint(query[0] + ranges[0], query[1] + ranges[1], 0.0);
        //down
        points->InsertNextPoint(query[0] - ranges[0], query[1] + ranges[1], 0.0);
        points->InsertNextPoint(query[0] + ranges[0], query[1] - ranges[1], 0.0);


        polygon->GetPointIds()->SetNumberOfIds(4); //make a quad
        polygon->GetPointIds()->SetId(0,0);
        polygon->GetPointIds()->SetId(1,1);
        polygon->GetPointIds()->SetId(2,2);
        polygon->GetPointIds()->SetId(3,3);

        // Add the polygon to a list of polygons
        polygons->InsertNextCell(polygon);

    }

    vtkSmartPointer<vtkPolyData> polygonPolyData =
            vtkSmartPointer<vtkPolyData>::New();
    polygonPolyData->SetPoints(points);
    polygonPolyData->SetPolys(polygons);

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polygonPolyData);

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(
            colors->GetColor3d("Grey").GetData());
    actor->GetProperty()->SetOpacity(0.7);

    return actor;
}

vtkSmartPointer<vtkActor> plot_splitting_lines(std::deque<std::vector<float>> & splitting_lines, int dimensionality, double * color){
    vtkSmartPointer<vtkPolyData> linesPolyData =
            vtkSmartPointer<vtkPolyData>::New();

    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();
    if (dimensionality == 3) {
        for (unsigned int i = 0; i < splitting_lines.size(); ++i) {
            points->InsertNextPoint(&splitting_lines[i][0]);
        }
    }
    else {
        for (unsigned int i = 0; i < splitting_lines.size(); ++i) {
            points->InsertNextPoint(splitting_lines[i][0], splitting_lines[i][1], 0.0);
        }
    }

    linesPolyData->SetPoints(points);

    vtkSmartPointer<vtkCellArray> lines =
            vtkSmartPointer<vtkCellArray>::New();

    for (unsigned int j = 0; j < splitting_lines.size(); j+=2) {
        vtkSmartPointer<vtkLine> line =
                vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, j);
        line->GetPointIds()->SetId(1, j+1);

        lines->InsertNextCell(line);
    }

    linesPolyData->SetLines(lines);

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();

    mapper->SetInputData(linesPolyData);

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color);

    return actor;
}

vtkSmartPointer<vtkActor> plot_bounding_box(std::vector<float> & box, int dimensionality, double * color){
    int MIN = 0;
    int MAX = 1;

    vtkSmartPointer<vtkPolyData> linesPolyData =
            vtkSmartPointer<vtkPolyData>::New();

    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();

    vtkSmartPointer<vtkCellArray> lines =
            vtkSmartPointer<vtkCellArray>::New();

    int num_points = 0;

    if (dimensionality == 2){
        std::vector<float> line_point1 = std::vector<float>(dimensionality + 1);
        std::vector<float> line_point2 = std::vector<float>(dimensionality + 1);
        std::vector<float> line_point3 = std::vector<float>(dimensionality + 1);
        std::vector<float> line_point4 = std::vector<float>(dimensionality + 1);

        // left up
        line_point1[0] = box[0 * 2 + MIN];
        line_point1[1] = box[1 * 2 + MAX];
        line_point1[2] = 0.0f;

        // right up
        line_point2[0] = box[0 * 2 + MAX];
        line_point2[1] = box[1 * 2 + MAX];
        line_point2[2] = 0.0f;

        // right down
        line_point3[0] = box[0 * 2 + MAX];
        line_point3[1] = box[1 * 2 + MIN];
        line_point3[2] = 0.0f;

        // left down
        line_point4[0] = box[0 * 2 + MIN];
        line_point4[1] = box[1 * 2 + MIN];
        line_point4[2] = 0.0f;

        // up line
        points->InsertNextPoint(&line_point1[0]);
        points->InsertNextPoint(&line_point2[0]);
        // right line
        points->InsertNextPoint(&line_point2[0]);
        points->InsertNextPoint(&line_point3[0]);
        // down line
        points->InsertNextPoint(&line_point3[0]);
        points->InsertNextPoint(&line_point4[0]);
        // left line
        points->InsertNextPoint(&line_point4[0]);
        points->InsertNextPoint(&line_point1[0]);

        num_points = 8;
    }
    else {
        // dimensionality is 3
        std::vector<float> line_point1 = std::vector<float>(dimensionality);
        std::vector<float> line_point2 = std::vector<float>(dimensionality);
        std::vector<float> line_point3 = std::vector<float>(dimensionality);
        std::vector<float> line_point4 = std::vector<float>(dimensionality);
        std::vector<float> line_point5 = std::vector<float>(dimensionality);
        std::vector<float> line_point6 = std::vector<float>(dimensionality);
        std::vector<float> line_point7 = std::vector<float>(dimensionality);
        std::vector<float> line_point8 = std::vector<float>(dimensionality);

        // back left up
        line_point1[0] = box[0 * 2 + MIN];
        line_point1[1] = box[1 * 2 + MAX];
        line_point1[2] = box[2 * 2 + MAX];

        // back right up
        line_point2[0] = box[0 * 2 + MAX];
        line_point2[1] = box[1 * 2 + MAX];
        line_point2[2] = box[2 * 2 + MAX];

        // back right down
        line_point3[0] = box[0 * 2 + MAX];
        line_point3[1] = box[1 * 2 + MAX];
        line_point3[2] = box[2 * 2 + MIN];

        // back left down
        line_point4[0] = box[0 * 2 + MIN];
        line_point4[1] = box[1 * 2 + MAX];
        line_point4[2] = box[2 * 2 + MIN];

        // front left up
        line_point5[0] = box[0 * 2 + MIN];
        line_point5[1] = box[1 * 2 + MIN];
        line_point5[2] = box[2 * 2 + MAX];

        // front right up
        line_point6[0] = box[0 * 2 + MAX];
        line_point6[1] = box[1 * 2 + MIN];
        line_point6[2] = box[2 * 2 + MAX];

        // front right down
        line_point7[0] = box[0 * 2 + MAX];
        line_point7[1] = box[1 * 2 + MIN];
        line_point7[2] = box[2 * 2 + MIN];

        // front left down
        line_point8[0] = box[0 * 2 + MIN];
        line_point8[1] = box[1 * 2 + MIN];
        line_point8[2] = box[2 * 2 + MIN];

        // back up line
        points->InsertNextPoint(&line_point1[0]);
        points->InsertNextPoint(&line_point2[0]);
        // back right line
        points->InsertNextPoint(&line_point2[0]);
        points->InsertNextPoint(&line_point3[0]);
        // back down line
        points->InsertNextPoint(&line_point3[0]);
        points->InsertNextPoint(&line_point4[0]);
        // back left line
        points->InsertNextPoint(&line_point4[0]);
        points->InsertNextPoint(&line_point1[0]);

        // front up line
        points->InsertNextPoint(&line_point5[0]);
        points->InsertNextPoint(&line_point6[0]);
        // front right line
        points->InsertNextPoint(&line_point6[0]);
        points->InsertNextPoint(&line_point7[0]);
        // front down line
        points->InsertNextPoint(&line_point7[0]);
        points->InsertNextPoint(&line_point8[0]);
        // front left line
        points->InsertNextPoint(&line_point8[0]);
        points->InsertNextPoint(&line_point5[0]);

        // front and back connecting splitting_lines
        // up left
        points->InsertNextPoint(&line_point1[0]);
        points->InsertNextPoint(&line_point5[0]);
        // up right
        points->InsertNextPoint(&line_point2[0]);
        points->InsertNextPoint(&line_point6[0]);
        // down right
        points->InsertNextPoint(&line_point3[0]);
        points->InsertNextPoint(&line_point7[0]);
        // down left
        points->InsertNextPoint(&line_point4[0]);
        points->InsertNextPoint(&line_point8[0]);

        num_points = 24;

    }

    linesPolyData->SetPoints(points);

    for (unsigned int j = 0; j < num_points; j+=2) {
        vtkSmartPointer<vtkLine> line =
                vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, j);
        line->GetPointIds()->SetId(1, j+1);

        lines->InsertNextCell(line);
    }

    linesPolyData->SetLines(lines);

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();

    mapper->SetInputData(linesPolyData);

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(color);

    return actor;

}

void show_visualization(std::vector<vtkSmartPointer<vtkActor>> & actors){
    // Create a renderer, render window
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    auto renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add the actors to the scene
    for (unsigned int i = 0; i < actors.size(); ++i) {
        renderer->AddActor(actors[i]);
    }

    renderer->SetBackground(1.0, 1.0, 1.0);

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();
}