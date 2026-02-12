
#pragma once
#pragma once
#include "Process.h"
#include "Network.h"
#include "MNISTReader.h"
#include <msclr/marshal_cppstd.h>

namespace MNISTProject {
    using namespace System;
    using namespace System::ComponentModel;
    using namespace System::Collections;
    using namespace System::Windows::Forms;
    using namespace System::Data;
    using namespace System::Drawing;
    using namespace System::IO;
    using namespace System::Windows::Forms::DataVisualization::Charting;



    public ref class MNISTForm : public System::Windows::Forms::Form
    {
    public:
        MNISTForm(void)
        {
            // FIRST: Initialize components pointer
            components = nullptr;

            // SECOND: Initialize the vector pointer
            layerNeurons = new std::vector<int>();

            // THIRD: Initialize UI components
            InitializeComponent();

            // THIRD: Set properties
            this->Text = L"MNIST Rakam Tanima Sistemi";

            // Initialize member variables
            numTrainingSamples = 0;
            numTestSamples = 0;
            inputDim = 784;
            outputDim = 10;
            trainingImages = nullptr;
            trainingLabels = nullptr;
            testImages = nullptr;
            testLabels = nullptr;
            dynamicWeights = nullptr;
            mean_params = nullptr;
            std_params = nullptr;
            isModelTrained = false;
            learningRate = 0.01f;
            momentumValue = 0.9f;
            useMomentum = true;
            maxEpochs = 100;
            numHiddenLayers = 2;
            isDrawing = false;

            // FOURTH: Initialize drawing canvas
            InitializeDrawingCanvas();

            // FIFTH: Setup background worker for training
            trainingWorker = gcnew System::ComponentModel::BackgroundWorker();
            trainingWorker->WorkerReportsProgress = true;
            trainingWorker->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &MNISTForm::TrainingWorker_DoWork);
            trainingWorker->ProgressChanged += gcnew System::ComponentModel::ProgressChangedEventHandler(this, &MNISTForm::TrainingWorker_ProgressChanged);
            trainingWorker->RunWorkerCompleted += gcnew System::ComponentModel::RunWorkerCompletedEventHandler(this, &MNISTForm::TrainingWorker_Completed);

            // LAST: Update layer config panel (after everything is ready)
            UpdateLayerConfigPanel();

        }
        

    protected:
        ~MNISTForm()
        {
            if (components) {
                delete components;
                components = nullptr;
            }

            if (layerNeurons) {
                delete layerNeurons;
                layerNeurons = nullptr;
            }

            if (trainingImages) {
                delete[] trainingImages;
                trainingImages = nullptr;
            }

            if (trainingLabels) {
                delete[] trainingLabels;
                trainingLabels = nullptr;
            }

            if (testImages) {
                delete[] testImages;
                testImages = nullptr;
            }

            if (testLabels) {
                delete[] testLabels;
                testLabels = nullptr;
            }

            if (dynamicWeights) {
                delete dynamicWeights;
                dynamicWeights = nullptr;
            }

           

            if (mean_params) {
                delete[] mean_params;
                mean_params = nullptr;
            }

            if (std_params) {
                delete[] std_params;
                std_params = nullptr;
            }
        }

    private:
        // Add BackgroundWorker to member variables
        System::ComponentModel::BackgroundWorker^ trainingWorker;

        // ... rest of your member variables ...
        System::Windows::Forms::GroupBox^ groupBoxData;
        System::Windows::Forms::GroupBox^ groupBoxModel;
        System::Windows::Forms::GroupBox^ groupBoxTest;
        System::Windows::Forms::GroupBox^ groupBoxDraw;
        System::Windows::Forms::Button^ btnLoadMNIST;
        System::Windows::Forms::Button^ btnTrainMLP;
        System::Windows::Forms::Button^ btnTestModel;
        System::Windows::Forms::Button^ btnClearCanvas;
        System::Windows::Forms::Button^ btnRecognize;
        System::Windows::Forms::Label^ lblTrainingCount;
        System::Windows::Forms::Label^ lblTestCount;
        System::Windows::Forms::Label^ lblEpochInfo;
        System::Windows::Forms::Label^ lblAccuracy;
        System::Windows::Forms::Label^ lblPrediction;
        System::Windows::Forms::Label^ lblConfidence;
        System::Windows::Forms::ProgressBar^ progressBarTraining;
        System::Windows::Forms::TextBox^ textBoxLog;
        System::Windows::Forms::PictureBox^ pictureBoxCanvas;
        System::Windows::Forms::PictureBox^ pictureBoxSamples;
        System::Windows::Forms::DataVisualization::Charting::Chart^ chartLoss;
        System::Windows::Forms::NumericUpDown^ numericEpochs;
        System::Windows::Forms::NumericUpDown^ numericLearningRate;
        System::Windows::Forms::NumericUpDown^ numericMomentum;
        System::Windows::Forms::ComboBox^ comboBoxLayers;
        System::Windows::Forms::Panel^ panelLayerConfig;
        System::Windows::Forms::CheckBox^ checkBoxMomentum;

        static MNISTForm^ g_activeForm = nullptr;
       


        int numTrainingSamples;
        int numTestSamples;
        int inputDim;
        int outputDim;
        float* trainingImages;
        int* trainingLabels;
        float* testImages;
        int* testLabels;
        DynamicMultiLayerWeights* dynamicWeights;
        float* mean_params;
        float* std_params;
        bool isModelTrained;
        float learningRate;
        float momentumValue;
        bool useMomentum;
        int maxEpochs;
        std::vector<int>* layerNeurons;
        int numHiddenLayers;
        bool isDrawing;
        Point lastPoint;
        Bitmap^ drawingBitmap;
        System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code

        void InitializeComponent(void)
        {
            System::Windows::Forms::DataVisualization::Charting::ChartArea^ chartArea1 =
                (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
            System::Windows::Forms::DataVisualization::Charting::Series^ series1 =
                (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
            System::Windows::Forms::DataVisualization::Charting::Title^ title1 =
                (gcnew System::Windows::Forms::DataVisualization::Charting::Title());

            this->groupBoxData = (gcnew System::Windows::Forms::GroupBox());
            this->groupBoxModel = (gcnew System::Windows::Forms::GroupBox());
            this->groupBoxTest = (gcnew System::Windows::Forms::GroupBox());
            this->groupBoxDraw = (gcnew System::Windows::Forms::GroupBox());

            this->btnLoadMNIST = (gcnew System::Windows::Forms::Button());
            this->btnTrainMLP = (gcnew System::Windows::Forms::Button());
            this->btnTestModel = (gcnew System::Windows::Forms::Button());
            this->btnClearCanvas = (gcnew System::Windows::Forms::Button());
            this->btnRecognize = (gcnew System::Windows::Forms::Button());

            this->lblTrainingCount = (gcnew System::Windows::Forms::Label());
            this->lblTestCount = (gcnew System::Windows::Forms::Label());
            this->lblEpochInfo = (gcnew System::Windows::Forms::Label());
            this->lblAccuracy = (gcnew System::Windows::Forms::Label());
            this->lblPrediction = (gcnew System::Windows::Forms::Label());
            this->lblConfidence = (gcnew System::Windows::Forms::Label());

            this->progressBarTraining = (gcnew System::Windows::Forms::ProgressBar());
            this->textBoxLog = (gcnew System::Windows::Forms::TextBox());
            this->pictureBoxCanvas = (gcnew System::Windows::Forms::PictureBox());
            this->pictureBoxSamples = (gcnew System::Windows::Forms::PictureBox());

            this->chartLoss = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());

            this->numericEpochs = (gcnew System::Windows::Forms::NumericUpDown());
            this->numericLearningRate = (gcnew System::Windows::Forms::NumericUpDown());
            this->numericMomentum = (gcnew System::Windows::Forms::NumericUpDown());
            this->comboBoxLayers = (gcnew System::Windows::Forms::ComboBox());
            this->panelLayerConfig = (gcnew System::Windows::Forms::Panel());
            this->checkBoxMomentum = (gcnew System::Windows::Forms::CheckBox());

            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxCanvas))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxSamples))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chartLoss))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericEpochs))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericLearningRate))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericMomentum))->BeginInit();

            this->groupBoxData->SuspendLayout();
            this->groupBoxModel->SuspendLayout();
            this->groupBoxTest->SuspendLayout();
            this->groupBoxDraw->SuspendLayout();
            this->SuspendLayout();

            // groupBoxData
            this->groupBoxData->Controls->Add(this->btnLoadMNIST);
            this->groupBoxData->Controls->Add(this->lblTrainingCount);
            this->groupBoxData->Controls->Add(this->lblTestCount);
            this->groupBoxData->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
            this->groupBoxData->Location = System::Drawing::Point(12, 12);
            this->groupBoxData->Name = L"groupBoxData";
            this->groupBoxData->Size = System::Drawing::Size(280, 140);
            this->groupBoxData->TabIndex = 0;
            this->groupBoxData->TabStop = false;
            this->groupBoxData->Text = L"📁 Veri Yönetimi";

            // btnLoadMNIST
            this->btnLoadMNIST->BackColor = System::Drawing::Color::FromArgb(79, 70, 229);
            this->btnLoadMNIST->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
            this->btnLoadMNIST->ForeColor = System::Drawing::Color::White;
            this->btnLoadMNIST->Location = System::Drawing::Point(15, 30);
            this->btnLoadMNIST->Name = L"btnLoadMNIST";
            this->btnLoadMNIST->Size = System::Drawing::Size(250, 40);
            this->btnLoadMNIST->TabIndex = 0;
            this->btnLoadMNIST->Text = L"MNIST Veri Setini Yükle";
            this->btnLoadMNIST->UseVisualStyleBackColor = false;
            this->btnLoadMNIST->Click += gcnew System::EventHandler(this, &MNISTForm::btnLoadMNIST_Click);

            // lblTrainingCount
            this->lblTrainingCount->AutoSize = true;
            this->lblTrainingCount->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9));
            this->lblTrainingCount->Location = System::Drawing::Point(15, 80);
            this->lblTrainingCount->Name = L"lblTrainingCount";
            this->lblTrainingCount->Size = System::Drawing::Size(150, 15);
            this->lblTrainingCount->TabIndex = 1;
            this->lblTrainingCount->Text = L"Eğitim Verisi: 0";

            // lblTestCount
            this->lblTestCount->AutoSize = true;
            this->lblTestCount->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9));
            this->lblTestCount->Location = System::Drawing::Point(15, 105);
            this->lblTestCount->Name = L"lblTestCount";
            this->lblTestCount->Size = System::Drawing::Size(150, 15);
            this->lblTestCount->TabIndex = 2;
            this->lblTestCount->Text = L"Test Verisi: 0";

            // groupBoxModel
            this->groupBoxModel->Controls->Add(this->btnTrainMLP);
            this->groupBoxModel->Controls->Add(this->comboBoxLayers);
            this->groupBoxModel->Controls->Add(this->panelLayerConfig);
            this->groupBoxModel->Controls->Add(this->numericEpochs);
            this->groupBoxModel->Controls->Add(this->numericLearningRate);
            this->groupBoxModel->Controls->Add(this->checkBoxMomentum);
            this->groupBoxModel->Controls->Add(this->numericMomentum);
            this->groupBoxModel->Controls->Add(this->progressBarTraining);
            this->groupBoxModel->Controls->Add(this->lblEpochInfo);
            this->groupBoxModel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
            this->groupBoxModel->Location = System::Drawing::Point(12, 158);
            this->groupBoxModel->Name = L"groupBoxModel";
            this->groupBoxModel->Size = System::Drawing::Size(280, 380);
            this->groupBoxModel->TabIndex = 1;
            this->groupBoxModel->TabStop = false;
            this->groupBoxModel->Text = L"🧠 MLP-FCN Model";

            // comboBoxLayers
            this->comboBoxLayers->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->comboBoxLayers->Items->AddRange(gcnew cli::array< System::Object^  >(4) { L"1", L"2", L"3", L"4" });
            this->comboBoxLayers->Location = System::Drawing::Point(15, 30);
            this->comboBoxLayers->Name = L"comboBoxLayers";
            this->comboBoxLayers->Size = System::Drawing::Size(250, 25);
            this->comboBoxLayers->TabIndex = 0;
            this->comboBoxLayers->SelectedIndex = 1;
            this->comboBoxLayers->SelectedIndexChanged += gcnew System::EventHandler(this, &MNISTForm::comboBoxLayers_SelectedIndexChanged);

            // panelLayerConfig
            this->panelLayerConfig->AutoScroll = true;
            this->panelLayerConfig->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->panelLayerConfig->Location = System::Drawing::Point(15, 60);
            this->panelLayerConfig->Name = L"panelLayerConfig";
            this->panelLayerConfig->Size = System::Drawing::Size(250, 100);
            this->panelLayerConfig->TabIndex = 1;

            // numericEpochs
            this->numericEpochs->Location = System::Drawing::Point(15, 170);
            this->numericEpochs->Maximum = 1000;
            this->numericEpochs->Minimum = 10;
            this->numericEpochs->Name = L"numericEpochs";
            this->numericEpochs->Size = System::Drawing::Size(120, 25);
            this->numericEpochs->TabIndex = 2;
            this->numericEpochs->Value = 100;

            // numericLearningRate
            this->numericLearningRate->DecimalPlaces = 4;
            this->numericLearningRate->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 262144 });
            this->numericLearningRate->Location = System::Drawing::Point(145, 170);
            this->numericLearningRate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->numericLearningRate->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 262144 });
            this->numericLearningRate->Name = L"numericLearningRate";
            this->numericLearningRate->Size = System::Drawing::Size(120, 25);
            this->numericLearningRate->TabIndex = 3;
            this->numericLearningRate->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 262144 });

            // checkBoxMomentum
            this->checkBoxMomentum->AutoSize = true;
            this->checkBoxMomentum->Checked = true;
            this->checkBoxMomentum->CheckState = System::Windows::Forms::CheckState::Checked;
            this->checkBoxMomentum->Location = System::Drawing::Point(15, 205);
            this->checkBoxMomentum->Name = L"checkBoxMomentum";
            this->checkBoxMomentum->Size = System::Drawing::Size(120, 23);
            this->checkBoxMomentum->TabIndex = 4;
            this->checkBoxMomentum->Text = L"Momentum";
            this->checkBoxMomentum->UseVisualStyleBackColor = true;

            // numericMomentum
            this->numericMomentum->DecimalPlaces = 2;
            this->numericMomentum->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 131072 });
            this->numericMomentum->Location = System::Drawing::Point(145, 205);
            this->numericMomentum->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 99, 0, 0, 131072 });
            this->numericMomentum->Name = L"numericMomentum";
            this->numericMomentum->Size = System::Drawing::Size(120, 25);
            this->numericMomentum->TabIndex = 5;
            this->numericMomentum->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 90, 0, 0, 131072 });

            // btnTrainMLP
            this->btnTrainMLP->BackColor = System::Drawing::Color::FromArgb(34, 197, 94);
            this->btnTrainMLP->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
            this->btnTrainMLP->ForeColor = System::Drawing::Color::White;
            this->btnTrainMLP->Location = System::Drawing::Point(15, 240);
            this->btnTrainMLP->Name = L"btnTrainMLP";
            this->btnTrainMLP->Size = System::Drawing::Size(250, 40);
            this->btnTrainMLP->TabIndex = 6;
            this->btnTrainMLP->Text = L"▶️ Eğitimi Başlat";
            this->btnTrainMLP->UseVisualStyleBackColor = false;
            this->btnTrainMLP->Click += gcnew System::EventHandler(this, &MNISTForm::btnTrainMLP_Click);

            // progressBarTraining
            this->progressBarTraining->Location = System::Drawing::Point(15, 290);
            this->progressBarTraining->Name = L"progressBarTraining";
            this->progressBarTraining->Size = System::Drawing::Size(250, 30);
            this->progressBarTraining->TabIndex = 7;

            // lblEpochInfo
            this->lblEpochInfo->AutoSize = true;
            this->lblEpochInfo->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9));
            this->lblEpochInfo->Location = System::Drawing::Point(15, 330);
            this->lblEpochInfo->Name = L"lblEpochInfo";
            this->lblEpochInfo->Size = System::Drawing::Size(100, 15);
            this->lblEpochInfo->TabIndex = 8;
            this->lblEpochInfo->Text = L"Epoch: 0/100";


            // groupBoxTest
            this->groupBoxTest->Controls->Add(this->btnTestModel);
            this->groupBoxTest->Controls->Add(this->lblAccuracy);
            this->groupBoxTest->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
            this->groupBoxTest->Location = System::Drawing::Point(12, 650);
            this->groupBoxTest->Name = L"groupBoxTest";
            this->groupBoxTest->Size = System::Drawing::Size(280, 120);
            this->groupBoxTest->TabIndex = 3;
            this->groupBoxTest->TabStop = false;
            this->groupBoxTest->Text = L"📊 Test";

            // btnTestModel
            this->btnTestModel->BackColor = System::Drawing::Color::FromArgb(249, 115, 22);
            this->btnTestModel->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
            this->btnTestModel->ForeColor = System::Drawing::Color::White;
            this->btnTestModel->Location = System::Drawing::Point(15, 30);
            this->btnTestModel->Name = L"btnTestModel";
            this->btnTestModel->Size = System::Drawing::Size(250, 40);
            this->btnTestModel->TabIndex = 0;
            this->btnTestModel->Text = L"Test Başlat (10K örnek)";
            this->btnTestModel->UseVisualStyleBackColor = false;
            this->btnTestModel->Click += gcnew System::EventHandler(this, &MNISTForm::btnTestModel_Click);

            // lblAccuracy
            this->lblAccuracy->AutoSize = true;
            this->lblAccuracy->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11, System::Drawing::FontStyle::Bold));
            this->lblAccuracy->ForeColor = System::Drawing::Color::FromArgb(34, 197, 94);
            this->lblAccuracy->Location = System::Drawing::Point(15, 80);
            this->lblAccuracy->Name = L"lblAccuracy";
            this->lblAccuracy->Size = System::Drawing::Size(120, 20);
            this->lblAccuracy->TabIndex = 1;
            this->lblAccuracy->Text = L"Doğruluk: --%";

            // groupBoxDraw
            this->groupBoxDraw->Controls->Add(this->pictureBoxCanvas);
            this->groupBoxDraw->Controls->Add(this->btnClearCanvas);
            this->groupBoxDraw->Controls->Add(this->btnRecognize);
            this->groupBoxDraw->Controls->Add(this->lblPrediction);
            this->groupBoxDraw->Controls->Add(this->lblConfidence);
            this->groupBoxDraw->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
            this->groupBoxDraw->Location = System::Drawing::Point(310, 12);
            this->groupBoxDraw->Name = L"groupBoxDraw";
            this->groupBoxDraw->Size = System::Drawing::Size(340, 480);
            this->groupBoxDraw->TabIndex = 4;
            this->groupBoxDraw->TabStop = false;
            this->groupBoxDraw->Text = L"✏️ Rakam Çiz ve Tanı";

            // pictureBoxCanvas
            this->pictureBoxCanvas->BackColor = System::Drawing::Color::White;
            this->pictureBoxCanvas->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->pictureBoxCanvas->Location = System::Drawing::Point(20, 30);
            this->pictureBoxCanvas->Name = L"pictureBoxCanvas";
            this->pictureBoxCanvas->Size = System::Drawing::Size(300, 300);
            this->pictureBoxCanvas->TabIndex = 0;
            this->pictureBoxCanvas->TabStop = false;
            this->pictureBoxCanvas->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &MNISTForm::pictureBoxCanvas_MouseDown);
            this->pictureBoxCanvas->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &MNISTForm::pictureBoxCanvas_MouseMove);
            this->pictureBoxCanvas->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &MNISTForm::pictureBoxCanvas_MouseUp);

            // btnClearCanvas
            this->btnClearCanvas->BackColor = System::Drawing::Color::FromArgb(239, 68, 68);
            this->btnClearCanvas->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
            this->btnClearCanvas->ForeColor = System::Drawing::Color::White;
            this->btnClearCanvas->Location = System::Drawing::Point(20, 340);
            this->btnClearCanvas->Name = L"btnClearCanvas";
            this->btnClearCanvas->Size = System::Drawing::Size(145, 35);
            this->btnClearCanvas->TabIndex = 1;
            this->btnClearCanvas->Text = L"🗑️ Temizle";
            this->btnClearCanvas->UseVisualStyleBackColor = false;
            this->btnClearCanvas->Click += gcnew System::EventHandler(this, &MNISTForm::btnClearCanvas_Click);

            // btnRecognize
            this->btnRecognize->BackColor = System::Drawing::Color::FromArgb(79, 70, 229);
            this->btnRecognize->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
            this->btnRecognize->ForeColor = System::Drawing::Color::White;
            this->btnRecognize->Location = System::Drawing::Point(175, 340);
            this->btnRecognize->Name = L"btnRecognize";
            this->btnRecognize->Size = System::Drawing::Size(145, 35);
            this->btnRecognize->TabIndex = 2;
            this->btnRecognize->Text = L"🔍 Tanı";
            this->btnRecognize->UseVisualStyleBackColor = false;
            this->btnRecognize->Click += gcnew System::EventHandler(this, &MNISTForm::btnRecognize_Click);

            // lblPrediction
            this->lblPrediction->Font = (gcnew System::Drawing::Font(L"Segoe UI", 48, System::Drawing::FontStyle::Bold));
            this->lblPrediction->ForeColor = System::Drawing::Color::FromArgb(79, 70, 229);
            this->lblPrediction->Location = System::Drawing::Point(20, 380);
            this->lblPrediction->Name = L"lblPrediction";
            this->lblPrediction->Size = System::Drawing::Size(300, 70);
            this->lblPrediction->TabIndex = 3;
            this->lblPrediction->Text = L"-";
            this->lblPrediction->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;

            // lblConfidence
            this->lblConfidence->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
            this->lblConfidence->Location = System::Drawing::Point(20, 450);
            this->lblConfidence->Name = L"lblConfidence";
            this->lblConfidence->Size = System::Drawing::Size(300, 20);
            this->lblConfidence->TabIndex = 4;
            this->lblConfidence->Text = L"Güven: --%";
            this->lblConfidence->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;

            // pictureBoxSamples
            this->pictureBoxSamples->BackColor = System::Drawing::Color::White;
            this->pictureBoxSamples->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->pictureBoxSamples->Location = System::Drawing::Point(310, 498);
            this->pictureBoxSamples->Name = L"pictureBoxSamples";
            this->pictureBoxSamples->Size = System::Drawing::Size(340, 272);
            this->pictureBoxSamples->TabIndex = 5;
            this->pictureBoxSamples->TabStop = false;

            // 
            // chartLoss
            // 
            chartArea1->Name = L"ChartArea1";
            this->chartLoss->ChartAreas->Add(chartArea1);
            this->chartLoss->Location = System::Drawing::Point(670, 12);
            this->chartLoss->Name = L"chartLoss";
            series1->BorderWidth = 2;
            series1->ChartArea = L"ChartArea1";
            series1->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Spline;
            series1->Color = System::Drawing::Color::Red;
            series1->Name = L"Loss";
            this->chartLoss->Series->Add(series1);
            this->chartLoss->Size = System::Drawing::Size(500, 350);
            this->chartLoss->TabIndex = 6;
            title1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
            title1->Name = L"Title1";
            title1->Text = L"📉 Eğitim Kaybı (Loss)";
            this->chartLoss->Titles->Add(title1);

            // textBoxLog
            this->textBoxLog->Font = (gcnew System::Drawing::Font(L"Consolas", 9));
            this->textBoxLog->Location = System::Drawing::Point(670, 370);
            this->textBoxLog->Multiline = true;
            this->textBoxLog->Name = L"textBoxLog";
            this->textBoxLog->ReadOnly = true;
            this->textBoxLog->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
            this->textBoxLog->Size = System::Drawing::Size(500, 400);
            this->textBoxLog->TabIndex = 7;

            // 
            // MNISTForm
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
            this->BackColor = System::Drawing::Color::FromArgb(243, 244, 246);
            this->ClientSize = System::Drawing::Size(1200, 800);
            this->Controls->Add(this->textBoxLog);
            this->Controls->Add(this->chartLoss);
            this->Controls->Add(this->pictureBoxSamples);
            this->Controls->Add(this->groupBoxDraw);
            this->Controls->Add(this->groupBoxTest);
            this->Controls->Add(this->groupBoxModel);
            this->Controls->Add(this->groupBoxData);
            this->Name = L"MNISTForm";
            this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
            this->Text = L"MNIST Rakam Tanıma Sistemi";

            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxCanvas))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxSamples))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chartLoss))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericEpochs))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericLearningRate))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericMomentum))->EndInit();

            this->groupBoxData->ResumeLayout(false);
            this->groupBoxData->PerformLayout();
            this->groupBoxModel->ResumeLayout(false);
            this->groupBoxModel->PerformLayout();
            this->groupBoxTest->ResumeLayout(false);
            this->groupBoxTest->PerformLayout();
            this->groupBoxDraw->ResumeLayout(false);
            this->ResumeLayout(false);
            this->PerformLayout();

            // Katman konfigürasyonunu başlat
            UpdateLayerConfigPanel();
        }

#pragma endregion


      
        void InitializeDrawingCanvas() {
            // 28x28'in 10 katı = 280x280 (daha büyük canvas)
            drawingBitmap = gcnew Bitmap(280, 280);
            Graphics^ g = Graphics::FromImage(drawingBitmap);
            g->Clear(Color::White);
            delete g;
        }

        
        void UpdateLayerConfigPanel() {
            // Safety check - make sure controls exist
            if (panelLayerConfig == nullptr || comboBoxLayers == nullptr) {
                return;
            }

            panelLayerConfig->Controls->Clear();

            // Safety check for layerNeurons
            if (layerNeurons == nullptr) {
                layerNeurons = new std::vector<int>();
            }

            layerNeurons->clear();
            numHiddenLayers = Convert::ToInt32(comboBoxLayers->Text);

            int yPos = 10;
            for (int i = 0; i < numHiddenLayers; i++) {
                Label^ lbl = gcnew Label();
                lbl->Text = "Katman " + (i + 1) + ":";
                lbl->Location = Point(10, yPos);
                lbl->AutoSize = true;

                ComboBox^ cb = gcnew ComboBox();
                cb->DropDownStyle = ComboBoxStyle::DropDownList;
                cb->Items->AddRange(gcnew cli::array<Object^> { L"32", L"64", L"128", L"256", L"512" });
                cb->SelectedIndex = (i == 0) ? 2 : 1;
                cb->Location = Point(80, yPos - 3);
                cb->Size = System::Drawing::Size(150, 25);
                cb->Name = "layer" + i;

                panelLayerConfig->Controls->Add(lbl);
                panelLayerConfig->Controls->Add(cb);
                yPos += 35;
            }
        }

        void LogMessage(String^ message) {
            textBoxLog->AppendText(DateTime::Now.ToString("HH:mm:ss") + " - " + message + "\r\n");
            textBoxLog->SelectionStart = textBoxLog->Text->Length;
            textBoxLog->ScrollToCaret();
            Application::DoEvents();
        }

        void comboBoxLayers_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
            UpdateLayerConfigPanel();
        }

        void btnLoadMNIST_Click(System::Object^ sender, System::EventArgs^ e) {
            FolderBrowserDialog^ folderDialog = gcnew FolderBrowserDialog();
            folderDialog->Description = L"MNIST veri klasorunu secin";

            if (folderDialog->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
                try {
                    LogMessage("MNIST verisi yukleniyor...");
                    String^ folder = folderDialog->SelectedPath;

                    std::string trainImagesPath = msclr::interop::marshal_as<std::string>(folder + "\\train-images.idx3-ubyte");
                    std::string trainLabelsPath = msclr::interop::marshal_as<std::string>(folder + "\\train-labels.idx1-ubyte");
                    std::string testImagesPath = msclr::interop::marshal_as<std::string>(folder + "\\t10k-images.idx3-ubyte");
                    std::string testLabelsPath = msclr::interop::marshal_as<std::string>(folder + "\\t10k-labels.idx1-ubyte");

                    int rows = 0, cols = 0;
                    int fullTrainCount = 0, fullTestCount = 0;
                    float* fullTrainImages = nullptr;
                    int* fullTrainLabels = nullptr;
                    float* fullTestImages = nullptr;
                    int* fullTestLabels = nullptr;

                    // ===== TÜM VERİYİ YÜKLE =====
                    if (!LoadMNISTImages(trainImagesPath.c_str(), fullTrainImages, fullTrainCount, rows, cols)) {
                        throw gcnew Exception("Egitim goruntuleri yuklenemedi!");
                    }

                    int loadedTrainingLabels = 0;
                    if (!LoadMNISTLabels(trainLabelsPath.c_str(), fullTrainLabels, loadedTrainingLabels)) {
                        if (fullTrainImages) delete[] fullTrainImages;
                        throw gcnew Exception("Egitim etiketleri yuklenemedi!");
                    }

                    int testRows = 0, testCols = 0;
                    if (!LoadMNISTImages(testImagesPath.c_str(), fullTestImages, fullTestCount, testRows, testCols)) {
                        if (fullTrainImages) delete[] fullTrainImages;
                        if (fullTrainLabels) delete[] fullTrainLabels;
                        throw gcnew Exception("Test goruntuleri yuklenemedi!");
                    }

                    int loadedTestLabels = 0;
                    if (!LoadMNISTLabels(testLabelsPath.c_str(), fullTestLabels, loadedTestLabels)) {
                        if (fullTrainImages) delete[] fullTrainImages;
                        if (fullTrainLabels) delete[] fullTrainLabels;
                        if (fullTestImages) delete[] fullTestImages;
                        throw gcnew Exception("Test etiketleri yuklenemedi!");
                    }

                    // ===== BALANCED DATASET OLUŞTUR =====
                    const int SAMPLES_PER_DIGIT_TRAIN = 100;  // Her rakamdan 100 eğitim
                    const int SAMPLES_PER_DIGIT_TEST = 10;    // Her rakamdan 10 test
                    const int NUM_DIGITS = 10;

                    int finalTrainCount = SAMPLES_PER_DIGIT_TRAIN * NUM_DIGITS;  // 1000 toplam
                    int finalTestCount = SAMPLES_PER_DIGIT_TEST * NUM_DIGITS;    // 100 toplam

                    // Yeni dizileri oluştur
                    float* balancedTrainImages = new float[finalTrainCount * inputDim];
                    int* balancedTrainLabels = new int[finalTrainCount];
                    float* balancedTestImages = new float[finalTestCount * inputDim];
                    int* balancedTestLabels = new int[finalTestCount];

                    // Her rakam için sayaç
                    int trainCountPerDigit[10] = { 0 };
                    int testCountPerDigit[10] = { 0 };
                    int trainInsertIdx = 0;
                    int testInsertIdx = 0;

                    // ===== EĞİTİM VERİSİNİ DOLDUR =====
                    for (int i = 0; i < fullTrainCount && trainInsertIdx < finalTrainCount; i++) {
                        int label = fullTrainLabels[i];

                        if (trainCountPerDigit[label] < SAMPLES_PER_DIGIT_TRAIN) {
                            // Görüntüyü kopyala
                            for (int j = 0; j < inputDim; j++) {
                                balancedTrainImages[trainInsertIdx * inputDim + j] =
                                    fullTrainImages[i * inputDim + j];
                            }
                            balancedTrainLabels[trainInsertIdx] = label;

                            trainCountPerDigit[label]++;
                            trainInsertIdx++;
                        }
                    }

                    // ===== TEST VERİSİNİ DOLDUR =====
                    for (int i = 0; i < fullTestCount && testInsertIdx < finalTestCount; i++) {
                        int label = fullTestLabels[i];

                        if (testCountPerDigit[label] < SAMPLES_PER_DIGIT_TEST) {
                            // Görüntüyü kopyala
                            for (int j = 0; j < inputDim; j++) {
                                balancedTestImages[testInsertIdx * inputDim + j] =
                                    fullTestImages[i * inputDim + j];
                            }
                            balancedTestLabels[testInsertIdx] = label;

                            testCountPerDigit[label]++;
                            testInsertIdx++;
                        }
                    }

                    // ===== ESKİ VERİLERİ TEMİZLE =====
                    delete[] fullTrainImages;
                    delete[] fullTrainLabels;
                    delete[] fullTestImages;
                    delete[] fullTestLabels;

                    if (trainingImages) delete[] trainingImages;
                    if (trainingLabels) delete[] trainingLabels;
                    if (testImages) delete[] testImages;
                    if (testLabels) delete[] testLabels;

                    // ===== YENİ VERİLERİ ATAR =====
                    trainingImages = balancedTrainImages;
                    trainingLabels = balancedTrainLabels;
                    testImages = balancedTestImages;
                    testLabels = balancedTestLabels;
                    numTrainingSamples = finalTrainCount;
                    numTestSamples = finalTestCount;
                    inputDim = rows * cols;

                    // ===== UI'Yİ GÜNCELLE =====
                    lblTrainingCount->Text = "Egitim: " + numTrainingSamples + " (her rakamdan " + SAMPLES_PER_DIGIT_TRAIN + ")";
                    lblTestCount->Text = "Test: " + numTestSamples + " (her rakamdan " + SAMPLES_PER_DIGIT_TEST + ")";

                    DrawSampleImages();

                    // Dağılımı logla
                    LogMessage("=== VERI DAGILIMI ===");
                    LogMessage("Egitim seti:");
                    for (int digit = 0; digit < 10; digit++) {
                        LogMessage("  Rakam " + digit + ": " + trainCountPerDigit[digit] + " ornek");
                    }
                    LogMessage("Test seti:");
                    for (int digit = 0; digit < 10; digit++) {
                        LogMessage("  Rakam " + digit + ": " + testCountPerDigit[digit] + " ornek");
                    }

                    LogMessage("Veri yuklendi!");
                    MessageBox::Show(
                        "Basarili!\n\nEgitim: " + numTrainingSamples + " ornek\nTest: " + numTestSamples + " ornek\n\nHer rakamdan esit sayida veri yuklendi.",
                        "MNIST Yuklendi",
                        MessageBoxButtons::OK,
                        MessageBoxIcon::Information
                    );
                }
                catch (Exception^ ex) {
                    LogMessage("Hata: " + ex->Message);
                    MessageBox::Show(ex->Message, "Hata", MessageBoxButtons::OK, MessageBoxIcon::Error);
                }
            }
        }

        void DrawSampleImages() {
            if (!trainingImages || numTrainingSamples == 0) return;
            Bitmap^ sampleBitmap = gcnew Bitmap(pictureBoxSamples->Width, pictureBoxSamples->Height);
            Graphics^ g = Graphics::FromImage(sampleBitmap);
            g->Clear(Color::White);
            int cellSize = pictureBoxSamples->Width / 10;

            for (int row = 0; row < 10; row++) {
                for (int col = 0; col < 10; col++) {
                    int sampleIdx = row * 10 + col;
                    if (sampleIdx >= numTrainingSamples) break;
                    int x = col * cellSize;
                    int y = row * cellSize;

                    for (int i = 0; i < 28; i++) {
                        for (int j = 0; j < 28; j++) {
                            float pixelValue = trainingImages[sampleIdx * 784 + i * 28 + j];
                            int gray = (int)(pixelValue * 255);
                            Color c = Color::FromArgb(gray, gray, gray);
                            int px = x + (j * cellSize / 28);
                            int py = y + (i * cellSize / 28);
                            if (px < sampleBitmap->Width && py < sampleBitmap->Height) {
                                sampleBitmap->SetPixel(px, py, c);
                            }
                        }
                    }
                }
            }
            pictureBoxSamples->Image = sampleBitmap;
            delete g;
        }

        void btnTrainMLP_Click(System::Object^ sender, System::EventArgs^ e) {
            if (numTrainingSamples == 0) {
                MessageBox::Show("Once veri yukleyin!", "Uyari", MessageBoxButtons::OK, MessageBoxIcon::Warning);
                return;
            }

            if (trainingWorker->IsBusy) {
                MessageBox::Show("Egitim zaten devam ediyor!", "Uyari", MessageBoxButtons::OK, MessageBoxIcon::Warning);
                return;
            }

            g_activeForm = this;

            maxEpochs = Convert::ToInt32(numericEpochs->Value);
            learningRate = Convert::ToSingle(numericLearningRate->Value);
            useMomentum = checkBoxMomentum->Checked;
            momentumValue = useMomentum ? Convert::ToSingle(numericMomentum->Value) : 0.0f;

            layerNeurons->clear();
            for each(Control ^ ctrl in panelLayerConfig->Controls) {
                if (ctrl->GetType() == ComboBox::typeid) {
                    layerNeurons->push_back(Convert::ToInt32(safe_cast<ComboBox^>(ctrl)->Text));
                }
            }

            LogMessage("=== EGITIM BASLIYOR ===");

            // FIX: Convert int vector to String array
            array<String^>^ layerStrings = gcnew array<String^>(static_cast<int>(layerNeurons->size()));
            for (int i = 0; i < static_cast<int>(layerNeurons->size()); i++) {
                layerStrings[i] = (*layerNeurons)[i].ToString();
            }

            LogMessage("Mimari: 784 -> " + String::Join(" -> ", layerStrings) + " -> 10");
            LogMessage("Epoch: " + maxEpochs + ", LR: " + learningRate.ToString("F4"));

            if (dynamicWeights) delete dynamicWeights;
            dynamicWeights = new DynamicMultiLayerWeights(inputDim, *layerNeurons, outputDim);

            if (mean_params) delete[] mean_params;
            if (std_params) delete[] std_params;
            mean_params = new float[inputDim];
            std_params = new float[inputDim];
            Z_Score_Parameters(trainingImages, numTrainingSamples, inputDim, mean_params, std_params);

            progressBarTraining->Value = 0;
            chartLoss->Series["Loss"]->Points->Clear();
            btnTrainMLP->Enabled = false;
            btnLoadMNIST->Enabled = false;

            trainingWorker->RunWorkerAsync();
        }

        void TrainingWorker_DoWork(System::Object^ sender, System::ComponentModel::DoWorkEventArgs^ e) {
            try {
                float* floatTargets = new float[numTrainingSamples];
                for (int i = 0; i < numTrainingSamples; i++) {
                    floatTargets[i] = (float)trainingLabels[i];
                }

                std::vector<float*> activations, derivatives, deltas;
                activations.push_back(new float[inputDim]);

                for (int l = 0; l < dynamicWeights->numLayers; l++) {
                    int size = dynamicWeights->layerSizes[l + 1];
                    activations.push_back(new float[size]);
                    derivatives.push_back(new float[size]);
                    deltas.push_back(new float[size]);
                }

                DynamicMomentumBuffers* momentumBuf = nullptr;
                if (useMomentum && momentumValue > 0.0f) {
                    momentumBuf = new DynamicMomentumBuffers(dynamicWeights, momentumValue);
                }

                for (int epoch = 0; epoch < maxEpochs; epoch++) {

                    if (trainingWorker->CancellationPending) {
                        e->Cancel = true;
                        break;
                    }

                    float avgError = TrainMultiLayer(
                        trainingImages,
                        floatTargets,
                        numTrainingSamples,
                        dynamicWeights,
                        learningRate,
                        mean_params,
                        std_params,
                        momentumBuf,
                        momentumValue,
                        activations,
                        derivatives,
                        deltas
                    );

                    int progress = ((epoch + 1) * 100) / maxEpochs;
                    array<Object^>^ progressData = gcnew array<Object^>(2);
                    progressData[0] = epoch + 1;
                    progressData[1] = avgError;

                    trainingWorker->ReportProgress(progress, progressData);
                }

                for (size_t i = 0; i < activations.size(); i++) delete[] activations[i];
                for (size_t i = 0; i < derivatives.size(); i++) delete[] derivatives[i];
                for (size_t i = 0; i < deltas.size(); i++) delete[] deltas[i];

                if (momentumBuf) delete momentumBuf;
                delete[] floatTargets;

            }
            catch (System::Exception^ ex) {
                e->Result = ex;
            }
        }


        void TrainingWorker_ProgressChanged(System::Object^ sender, System::ComponentModel::ProgressChangedEventArgs^ e) {
            // Update UI from UI thread
            progressBarTraining->Value = e->ProgressPercentage;

            array<Object^>^ data = safe_cast<array<Object^>^>(e->UserState);
            int epoch = safe_cast<int>(data[0]);
            float avgError = safe_cast<float>(data[1]);

            lblEpochInfo->Text = "Epoch: " + epoch + "/" + maxEpochs;
            chartLoss->Series["Loss"]->Points->AddXY(epoch, avgError);

            if (epoch % 5 == 0 || epoch == maxEpochs) {
                LogMessage("Epoch " + epoch + "/" + maxEpochs + " - Loss: " + avgError.ToString("F4"));
            }
        }

        void TrainingWorker_Completed(System::Object^ sender, System::ComponentModel::RunWorkerCompletedEventArgs^ e) {
            btnTrainMLP->Enabled = true;
            btnLoadMNIST->Enabled = true;

            if (e->Error != nullptr) {
                LogMessage("HATA: " + e->Error->Message);
                MessageBox::Show("Egitim hatasi: " + e->Error->Message, "Hata",
                    MessageBoxButtons::OK, MessageBoxIcon::Error);
                isModelTrained = false;  // IMPORTANT: Set to false on error
            }
            else if (e->Result != nullptr && e->Result->GetType() == System::Exception::typeid) {
                System::Exception^ ex = safe_cast<System::Exception^>(e->Result);
                LogMessage("HATA: " + ex->Message);
                MessageBox::Show("Egitim hatasi: " + ex->Message, "Hata",
                    MessageBoxButtons::OK, MessageBoxIcon::Error);
                isModelTrained = false;  // IMPORTANT: Set to false on error
            }
            else {
                isModelTrained = true;  // CRITICAL: Set to true on success
                LogMessage("EGITIM TAMAM!");
                MessageBox::Show("Egitim tamamlandi!", "Basarili",
                    MessageBoxButtons::OK, MessageBoxIcon::Information);
            }
        }


        void btnTestModel_Click(System::Object^ sender, System::EventArgs^ e) {
            if (!isModelTrained) {
                MessageBox::Show("Once model egitin!", "Uyari", MessageBoxButtons::OK, MessageBoxIcon::Warning);
                return;
            }

            LogMessage("Test basliyor...");
            btnTestModel->Enabled = false;

            int correct = 0;
            DynamicMultiLayerConfig config;
            config.inputDim = inputDim;
            config.outputDim = outputDim;
            config.hiddenLayers = *layerNeurons;

            for (int i = 0; i < numTestSamples; i++) {
                int prediction = TestDynamicMultiLayer(&testImages[i * inputDim], dynamicWeights, config, mean_params, std_params);
                if (prediction == testLabels[i]) correct++;
                if (i % 1000 == 0) Application::DoEvents();
            }

            float accuracy = (correct / (float)numTestSamples) * 100.0f;
            lblAccuracy->Text = "Dogruluk: " + accuracy.ToString("F2") + "%";
            lblAccuracy->ForeColor = (accuracy > 90) ? Color::Green : Color::Orange;
            LogMessage("Test tamam! Dogruluk: " + accuracy.ToString("F2") + "%");
            btnTestModel->Enabled = true;
            MessageBox::Show("Dogruluk: " + accuracy.ToString("F2") + "%", "Test Sonucu", MessageBoxButtons::OK, MessageBoxIcon::Information);
        }

        void pictureBoxCanvas_MouseDown(System::Object^ sender, MouseEventArgs^ e) {
            isDrawing = true;
            lastPoint = e->Location;
        }

        void pictureBoxCanvas_MouseMove(System::Object^ sender, MouseEventArgs^ e) {
            if (!isDrawing) return;
            Graphics^ g = Graphics::FromImage(drawingBitmap);
            Pen^ pen = gcnew Pen(Color::Black, 20);
            pen->StartCap = System::Drawing::Drawing2D::LineCap::Round;
            pen->EndCap = System::Drawing::Drawing2D::LineCap::Round;
            g->DrawLine(pen, lastPoint, e->Location);
            lastPoint = e->Location;
            pictureBoxCanvas->Image = drawingBitmap;
            pictureBoxCanvas->Refresh();
            delete pen;
            delete g;
        }

        void pictureBoxCanvas_MouseUp(System::Object^ sender, MouseEventArgs^ e) {
            isDrawing = false;
        }

        void btnClearCanvas_Click(System::Object^ sender, System::EventArgs^ e) {
            Graphics^ g = Graphics::FromImage(drawingBitmap);
            g->Clear(Color::White);
            delete g;
            pictureBoxCanvas->Image = drawingBitmap;
            pictureBoxCanvas->Refresh();
            lblPrediction->Text = "-";
            lblConfidence->Text = "Guven: --%";
        }

        float Clamp01(float x)
        {
            if (x < 0.0f) return 0.0f;
            if (x > 1.0f) return 1.0f;
            return x;
        }


        void btnRecognize_Click(System::Object^ sender, System::EventArgs^ e) {
            if (!isModelTrained) {
                MessageBox::Show("Önce model eğitin!", "Uyarı", MessageBoxButtons::OK, MessageBoxIcon::Warning);
                return;
            }

            try {
                // ===== 1. Canvas'ı 28x28'e boyutlandır =====
                Bitmap^ resized = gcnew Bitmap(28, 28);
                Graphics^ g = Graphics::FromImage(resized);
                g->InterpolationMode = System::Drawing::Drawing2D::InterpolationMode::HighQualityBicubic;
                g->SmoothingMode = System::Drawing::Drawing2D::SmoothingMode::AntiAlias;
                g->Clear(Color::White);
                g->DrawImage(drawingBitmap, 0, 0, 28, 28);
                delete g;

                // ===== 2. Piksel verisini float array'e çevir =====
                float* rawInput = new float[inputDim];

                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        Color pixel = resized->GetPixel(x, y);
                        int gray = pixel.R;
                        // MNIST format: 0.0 = beyaz (arka plan), 1.0 = siyah (rakam)
                        rawInput[y * 28 + x] = (255.0f - gray) / 255.0f;
                    }
                }

                // ===== 3. Boş canvas kontrolü =====
                float totalMass = 0.0f;
                for (int i = 0; i < inputDim; i++) {
                    totalMass += rawInput[i];
                }

                if (totalMass < 5.0f) { // Eşik değeri artırıldı
                    LogMessage("UYARI: Canvas çok az çizim içeriyor!");
                    lblPrediction->Text = "?";
                    lblConfidence->Text = "Çizim yok";
                    delete[] rawInput;
                    delete resized;
                    return;
                }

                // ===== 4. Kütle merkezi hesapla =====
                float centerX = 0.0f, centerY = 0.0f;
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int idx = y * 28 + x;
                        float pixel = rawInput[idx];
                        centerX += x * pixel;
                        centerY += y * pixel;
                    }
                }
                centerX /= totalMass;
                centerY /= totalMass;

                // ===== 5. Çizimi merkeze kaydır =====
                float* centeredInput = new float[inputDim];
                std::fill(centeredInput, centeredInput + inputDim, 0.0f);

                int shiftX = (int)Math::Round(13.5f - centerX);
                int shiftY = (int)Math::Round(13.5f - centerY);

                // Kayma sınırlaması
                shiftX = std::max(-5, std::min(5, shiftX));
                shiftY = std::max(-5, std::min(5, shiftY));

                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int srcX = x - shiftX;
                        int srcY = y - shiftY;
                        if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
                            centeredInput[y * 28 + x] = rawInput[srcY * 28 + srcX];
                        }
                    }
                }

                LogMessage("Kütle merkezi: (" + centerX.ToString("F1") + ", " +
                    centerY.ToString("F1") + "), Kayma: (" + shiftX + ", " + shiftY + ")");

                // ===== 6. Forward pass için aktivasyonlar =====
                std::vector<float*> activations;
                activations.push_back(new float[inputDim]);

                for (int l = 0; l < dynamicWeights->numLayers; l++) {
                    activations.push_back(new float[dynamicWeights->layerSizes[l + 1]]);
                }

                // Z-score normalizasyonu
                for (int i = 0; i < inputDim; i++) {
                    float std_val = (std_params[i] < 1e-8f) ? 1.0f : std_params[i];
                    activations[0][i] = (centeredInput[i] - mean_params[i]) / std_val;
                }

                // ===== 7. Forward pass =====
                for (int l = 0; l < dynamicWeights->numLayers; l++) {
                    int inputSize = dynamicWeights->layerSizes[l];
                    int outputSize = dynamicWeights->layerSizes[l + 1];

                    for (int j = 0; j < outputSize; j++) {
                        float net = dynamicWeights->b[l][j];
                        for (int i = 0; i < inputSize; i++) {
                            net += dynamicWeights->W[l][j * inputSize + i] * activations[l][i];
                        }
                        activations[l + 1][j] = Tanh(net);
                    }
                }

                float* outputs = activations.back();

                // ===== 8) Softmax YOK: Top1 + Top2 + Margin ile tahmin & güven =====
                int prediction = 0;
                float top1 = outputs[0];
                float top2 = -1e9f;

                for (int i = 1; i < outputDim; i++) {
                    float v = outputs[i];
                    if (v > top1) {
                        top2 = top1;
                        top1 = v;
                        prediction = i;
                    }
                    else if (v > top2) {
                        top2 = v;
                    }
                }

                // margin: Top1 - Top2 (tanh çıktıları için en stabil güven ölçüsü)
                float margin = top1 - top2;

                // [0..1] aralığına softmax'sız güven skoru:
                // - margin maksimum ~2 (1 - (-1))
                // - top1 [-1..1] => (top1+1)/2 ile [0..1] yapıyoruz
                

                float margin01 = Clamp01(margin / 2.0f);
                float top101 = Clamp01((top1 + 1.0f) / 2.0f);


                // İstersen ağırlıklarla oynarsın; bu kombinasyon UI için çok stabil:
                float conf01 = Clamp01(0.70f * margin01 + 0.30f * top101);
                float confPct = conf01 * 100.0f;

                // ===== 9) Eşik (emin değilse uyar) =====
                const float MARGIN_TH = 0.15f; // kararsızlık eşiği (gerekirse 0.10-0.25 arası dene)
                const float TOP1_TH = 0.20f; // çıktı çok düşükse (tanh) yine şüpheli

                lblPrediction->Text = prediction.ToString();

                if (margin < MARGIN_TH || top1 < TOP1_TH) {
                    lblConfidence->Text = "Emin değil: " + confPct.ToString("F1") + "%";
                    lblConfidence->ForeColor = Color::OrangeRed;
                }
                else {
                    lblConfidence->Text = "Güven: " + confPct.ToString("F1") + "%";
                    if (confPct >= 80.0f)      lblConfidence->ForeColor = Color::DarkGreen;
                    else if (confPct >= 60.0f) lblConfidence->ForeColor = Color::Green;
                    else if (confPct >= 40.0f) lblConfidence->ForeColor = Color::Orange;
                    else                       lblConfidence->ForeColor = Color::Red;
                }

                // ===== 10) Log (softmax yok, raw skor + margin göster) =====
                LogMessage("=== TAHMİN (Softmax Yok) ===");
                LogMessage("Rakam: " + prediction +
                    " | top1=" + top1.ToString("F3") +
                    " | top2=" + top2.ToString("F3") +
                    " | margin=" + margin.ToString("F3") +
                    " | güven=" + confPct.ToString("F1") + "%");

                // Top 3: artık probs değil, outputs'a göre
                int indices[10];
                for (int i = 0; i < 10; i++) indices[i] = i;

                for (int i = 0; i < 3; i++) {
                    for (int j = i + 1; j < 10; j++) {
                        if (outputs[indices[j]] > outputs[indices[i]]) {
                            int temp = indices[i];
                            indices[i] = indices[j];
                            indices[j] = temp;
                        }
                    }
                }

                LogMessage("Top 3 (raw tanh skor):");
                for (int i = 0; i < 3; i++) {
                    int idx = indices[i];
                    LogMessage("  " + (i + 1) + ". Rakam " + idx + " → " + outputs[idx].ToString("F3"));
                }


                // ===== 11. Temizlik =====
                for (size_t i = 0; i < activations.size(); i++) {
                    delete[] activations[i];
                }
                delete[] rawInput;
                delete[] centeredInput;
                delete resized;
            }
            catch (Exception^ ex) {
                LogMessage("HATA: " + ex->Message);
                MessageBox::Show("Tanıma hatası: " + ex->Message, "Hata",
                    MessageBoxButtons::OK, MessageBoxIcon::Error);
            }
        }

    };
    
    }