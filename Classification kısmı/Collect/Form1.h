#pragma once
#include "Process.h"
#include "Network.h"
#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>



namespace CppCLRWinformsProjekt {

    using namespace System;
    using namespace System::ComponentModel;
    using namespace System::Collections;
    using namespace System::Windows::Forms;
    using namespace System::Data;
    using namespace System::Drawing;
    using namespace System::IO;
    using namespace System::Windows::Forms::DataVisualization::Charting;

    public ref class Form1 : public System::Windows::Forms::Form
    {
    public:
        Form1(void)
        {
            InitializeComponent();
            layerNeurons = new std::vector<int>();

            this->Text = L"YAPAY SİNİR AĞLARI";

        }

    protected:

        ~Form1()
        {
            if (components) delete components;
            if (layerNeurons) delete layerNeurons;

           
            if (Samples) delete[] Samples;
            if (targets) delete[] targets;
            if (Weights) delete[] Weights;
            if (bias) delete[] bias;
            if (mean_params) delete[] mean_params;
            if (std_params) delete[] std_params;
            if (dynamicWeights) delete dynamicWeights;
            
        }


    private: System::Windows::Forms::PictureBox^ pictureBox1;
    protected:
    private: System::Windows::Forms::GroupBox^ groupBox1;
    private: System::Windows::Forms::Button^ Set_Net;

    private: System::Windows::Forms::Label^ label1;
    private: System::Windows::Forms::ComboBox^ ClassCountBox;

    private: System::Windows::Forms::GroupBox^ groupBox2;
    private: System::Windows::Forms::Label^ label2;
    private: System::Windows::Forms::ComboBox^ ClassNoBox;

    private: System::Windows::Forms::Label^ label3;

    private: System::Windows::Forms::Label^ labelMomentum;
    private: System::Windows::Forms::TextBox^ MomentumTextBox;
    private: System::Windows::Forms::CheckBox^ UseMomentumCheckBox;


    private:
        int class_count = 0, numSample = 0, inputDim = 2;
        float* Samples, * targets, * Weights, * bias;

        bool isMultiLayer = false;
        bool useDynamicLayers = false;
        int numHiddenLayers = 1;
        std::vector<int>* layerNeurons;

        DynamicMultiLayerWeights* dynamicWeights = nullptr;

        float* mean_params = nullptr;
        float* std_params = nullptr;

        float momentumValue = 0.9f;
        bool useMomentum = true;

        bool isRegression = false; 

        RegressionWeights* regressionWeights = nullptr;
        DynamicRegressionWeights* dynamicRegressionWeights = nullptr;

        float mean_y = 0.0f;
        float std_y = 1.0f;

    private: System::Windows::Forms::MenuStrip^ menuStrip1;
    private: System::Windows::Forms::ToolStripMenuItem^ fileToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^ readDataToolStripMenuItem;
    private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
    private: System::Windows::Forms::TextBox^ textBox1;
    private: System::Windows::Forms::ToolStripMenuItem^ saveDataToolStripMenuItem;
    private: System::Windows::Forms::SaveFileDialog^ saveFileDialog1;
    private: System::Windows::Forms::ToolStripMenuItem^ processToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^ trainingToolStripMenuItem;
    private: System::Windows::Forms::ToolStripMenuItem^ testingToolStripMenuItem;
    private: System::Windows::Forms::DataVisualization::Charting::Chart^ chartEpoch;
    private: System::Windows::Forms::CheckBox^ RegressionCheckBox;

    private:
        System::Windows::Forms::Label^ labelNumLayers;
        System::Windows::Forms::ComboBox^ NumLayersBox;
        System::Windows::Forms::Panel^ layerNeuronsPanel;
        System::Windows::Forms::CheckBox^ DynamicLayersCheckBox;

    private: System::Windows::Forms::Label^ labelLearningRate;
    private: System::Windows::Forms::TextBox^ LearningRateTextBox;
    private: System::Windows::Forms::Button^ btnTum;
    private: System::Windows::Forms::Button^ btnSon;
private: System::Windows::Forms::TextBox^ MaxEpochTextBox;

private: System::Windows::Forms::Label^ labelHataTekrarı;

private: System::Windows::Forms::Label^ labelHataDegeri;


private: System::Windows::Forms::Label^ labelEpoch;
private: System::Windows::Forms::TextBox^ PatienceTextBox;


private: System::Windows::Forms::TextBox^ EarlyStopTextBox;



        System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
        
        void InitializeComponent(void)
        {
            System::Windows::Forms::DataVisualization::Charting::ChartArea^ chartArea1 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
            System::Windows::Forms::DataVisualization::Charting::Series^ series1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
            System::Windows::Forms::DataVisualization::Charting::Title^ title1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Title());
            this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
            this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
            this->PatienceTextBox = (gcnew System::Windows::Forms::TextBox());
            this->EarlyStopTextBox = (gcnew System::Windows::Forms::TextBox());
            this->MaxEpochTextBox = (gcnew System::Windows::Forms::TextBox());
            this->labelHataTekrarı = (gcnew System::Windows::Forms::Label());
            this->labelHataDegeri = (gcnew System::Windows::Forms::Label());
            this->labelEpoch = (gcnew System::Windows::Forms::Label());
            this->label1 = (gcnew System::Windows::Forms::Label());
            this->ClassCountBox = (gcnew System::Windows::Forms::ComboBox());
            this->DynamicLayersCheckBox = (gcnew System::Windows::Forms::CheckBox());
            this->labelNumLayers = (gcnew System::Windows::Forms::Label());
            this->NumLayersBox = (gcnew System::Windows::Forms::ComboBox());
            this->layerNeuronsPanel = (gcnew System::Windows::Forms::Panel());
            this->UseMomentumCheckBox = (gcnew System::Windows::Forms::CheckBox());
            this->labelMomentum = (gcnew System::Windows::Forms::Label());
            this->MomentumTextBox = (gcnew System::Windows::Forms::TextBox());
            this->labelLearningRate = (gcnew System::Windows::Forms::Label());
            this->LearningRateTextBox = (gcnew System::Windows::Forms::TextBox());
            this->Set_Net = (gcnew System::Windows::Forms::Button());
            this->RegressionCheckBox = (gcnew System::Windows::Forms::CheckBox());
            this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
            this->btnTum = (gcnew System::Windows::Forms::Button());
            this->btnSon = (gcnew System::Windows::Forms::Button());
            this->label2 = (gcnew System::Windows::Forms::Label());
            this->ClassNoBox = (gcnew System::Windows::Forms::ComboBox());
            this->label3 = (gcnew System::Windows::Forms::Label());
            this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
            this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->readDataToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->saveDataToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->processToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->trainingToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->testingToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->chartEpoch = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
            this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
            this->textBox1 = (gcnew System::Windows::Forms::TextBox());
            this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
            this->groupBox1->SuspendLayout();
            this->groupBox2->SuspendLayout();
            this->menuStrip1->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chartEpoch))->BeginInit();
            this->SuspendLayout();
            // 
            // pictureBox1
            // 
            this->pictureBox1->BackColor = System::Drawing::SystemColors::ButtonHighlight;
            this->pictureBox1->Location = System::Drawing::Point(13, 35);
            this->pictureBox1->Name = L"pictureBox1";
            this->pictureBox1->Size = System::Drawing::Size(802, 578);
            this->pictureBox1->TabIndex = 0;
            this->pictureBox1->TabStop = false;
            this->pictureBox1->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &Form1::pictureBox1_Paint);
            this->pictureBox1->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseClick);
            // 
            // groupBox1
            // 
            this->groupBox1->BackColor = System::Drawing::SystemColors::ButtonHighlight;
            this->groupBox1->Controls->Add(this->PatienceTextBox);
            this->groupBox1->Controls->Add(this->EarlyStopTextBox);
            this->groupBox1->Controls->Add(this->MaxEpochTextBox);
            this->groupBox1->Controls->Add(this->labelHataTekrarı);
            this->groupBox1->Controls->Add(this->labelHataDegeri);
            this->groupBox1->Controls->Add(this->labelEpoch);
            this->groupBox1->Controls->Add(this->label1);
            this->groupBox1->Controls->Add(this->ClassCountBox);
            this->groupBox1->Controls->Add(this->DynamicLayersCheckBox);
            this->groupBox1->Controls->Add(this->labelNumLayers);
            this->groupBox1->Controls->Add(this->NumLayersBox);
            this->groupBox1->Controls->Add(this->layerNeuronsPanel);
            this->groupBox1->Controls->Add(this->UseMomentumCheckBox);
            this->groupBox1->Controls->Add(this->labelMomentum);
            this->groupBox1->Controls->Add(this->MomentumTextBox);
            this->groupBox1->Controls->Add(this->labelLearningRate);
            this->groupBox1->Controls->Add(this->LearningRateTextBox);
            this->groupBox1->Controls->Add(this->Set_Net);
            this->groupBox1->Controls->Add(this->RegressionCheckBox);
            this->groupBox1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold));
            this->groupBox1->Location = System::Drawing::Point(869, 50);
            this->groupBox1->Name = L"groupBox1";
            this->groupBox1->Size = System::Drawing::Size(200, 484);
            this->groupBox1->TabIndex = 1;
            this->groupBox1->TabStop = false;
            this->groupBox1->Text = L"Network Architecture";
            // 
            // PatienceTextBox
            // 
            this->PatienceTextBox->Location = System::Drawing::Point(5, 360);
            this->PatienceTextBox->Name = L"PatienceTextBox";
            this->PatienceTextBox->Size = System::Drawing::Size(80, 20);
            this->PatienceTextBox->TabIndex = 19;
            this->PatienceTextBox->Text = L"100";
            // 
            // EarlyStopTextBox
            // 
            this->EarlyStopTextBox->Location = System::Drawing::Point(5, 334);
            this->EarlyStopTextBox->Name = L"EarlyStopTextBox";
            this->EarlyStopTextBox->Size = System::Drawing::Size(80, 20);
            this->EarlyStopTextBox->TabIndex = 19;
            this->EarlyStopTextBox->Text = L"0.001";
            // 
            // MaxEpochTextBox
            // 
            this->MaxEpochTextBox->Location = System::Drawing::Point(5, 308);
            this->MaxEpochTextBox->Name = L"MaxEpochTextBox";
            this->MaxEpochTextBox->Size = System::Drawing::Size(80, 20);
            this->MaxEpochTextBox->TabIndex = 18;
            this->MaxEpochTextBox->Text = L"2000";
            // 
            // labelHataTekrarı
            // 
            this->labelHataTekrarı->AutoSize = true;
            this->labelHataTekrarı->Location = System::Drawing::Point(98, 363);
            this->labelHataTekrarı->Name = L"labelHataTekrarı";
            this->labelHataTekrarı->Size = System::Drawing::Size(78, 13);
            this->labelHataTekrarı->TabIndex = 17;
            this->labelHataTekrarı->Text = L"Hata Tekrarı";
            // 
            // labelHataDegeri
            // 
            this->labelHataDegeri->AutoSize = true;
            this->labelHataDegeri->Location = System::Drawing::Point(99, 337);
            this->labelHataDegeri->Name = L"labelHataDegeri";
            this->labelHataDegeri->Size = System::Drawing::Size(75, 13);
            this->labelHataDegeri->TabIndex = 16;
            this->labelHataDegeri->Text = L"Hata Değeri";
            // 
            // labelEpoch
            // 
            this->labelEpoch->AutoSize = true;
            this->labelEpoch->Location = System::Drawing::Point(99, 311);
            this->labelEpoch->Name = L"labelEpoch";
            this->labelEpoch->Size = System::Drawing::Size(80, 13);
            this->labelEpoch->TabIndex = 15;
            this->labelEpoch->Text = L"Epoch Sayısı";
            // 
            // label1
            // 
            this->label1->AutoSize = true;
            this->label1->Location = System::Drawing::Point(98, 23);
            this->label1->Name = L"label1";
            this->label1->Size = System::Drawing::Size(69, 13);
            this->label1->TabIndex = 1;
            this->label1->Text = L"Sınıf Sayısı";
            // 
            // ClassCountBox
            // 
            this->ClassCountBox->FormattingEnabled = true;
            this->ClassCountBox->Items->AddRange(gcnew cli::array< System::Object^  >(6) { L"2", L"3", L"4", L"5", L"6", L"7" });
            this->ClassCountBox->Location = System::Drawing::Point(10, 20);
            this->ClassCountBox->Name = L"ClassCountBox";
            this->ClassCountBox->Size = System::Drawing::Size(82, 21);
            this->ClassCountBox->TabIndex = 0;
            this->ClassCountBox->Text = L"2";
            // 
            // DynamicLayersCheckBox
            // 
            this->DynamicLayersCheckBox->AutoSize = true;
            this->DynamicLayersCheckBox->Location = System::Drawing::Point(10, 50);
            this->DynamicLayersCheckBox->Name = L"DynamicLayersCheckBox";
            this->DynamicLayersCheckBox->Size = System::Drawing::Size(140, 17);
            this->DynamicLayersCheckBox->TabIndex = 3;
            this->DynamicLayersCheckBox->Text = L"Dynamic Multi-Layer";
            this->DynamicLayersCheckBox->UseVisualStyleBackColor = true;
            this->DynamicLayersCheckBox->CheckedChanged += gcnew System::EventHandler(this, &Form1::DynamicLayersCheckBox_CheckedChanged);
            // 
            // labelNumLayers
            // 
            this->labelNumLayers->AutoSize = true;
            this->labelNumLayers->Location = System::Drawing::Point(98, 75);
            this->labelNumLayers->Name = L"labelNumLayers";
            this->labelNumLayers->Size = System::Drawing::Size(88, 13);
            this->labelNumLayers->TabIndex = 4;
            this->labelNumLayers->Text = L"Hidden Layers";
            this->labelNumLayers->Visible = false;
            // 
            // NumLayersBox
            // 
            this->NumLayersBox->FormattingEnabled = true;
            this->NumLayersBox->Items->AddRange(gcnew cli::array< System::Object^  >(5) { L"1", L"2", L"3", L"4", L"5" });
            this->NumLayersBox->Location = System::Drawing::Point(10, 72);
            this->NumLayersBox->Name = L"NumLayersBox";
            this->NumLayersBox->Size = System::Drawing::Size(82, 21);
            this->NumLayersBox->TabIndex = 5;
            this->NumLayersBox->Text = L"1";
            this->NumLayersBox->Visible = false;
            this->NumLayersBox->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::NumLayersBox_SelectedIndexChanged);
            // 
            // layerNeuronsPanel
            // 
            this->layerNeuronsPanel->AutoScroll = true;
            this->layerNeuronsPanel->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->layerNeuronsPanel->Location = System::Drawing::Point(10, 100);
            this->layerNeuronsPanel->Name = L"layerNeuronsPanel";
            this->layerNeuronsPanel->Size = System::Drawing::Size(180, 150);
            this->layerNeuronsPanel->TabIndex = 6;
            this->layerNeuronsPanel->Visible = false;
            // 
            // UseMomentumCheckBox
            // 
            this->UseMomentumCheckBox->AutoSize = true;
            this->UseMomentumCheckBox->Checked = true;
            this->UseMomentumCheckBox->CheckState = System::Windows::Forms::CheckState::Checked;
            this->UseMomentumCheckBox->Location = System::Drawing::Point(6, 385);
            this->UseMomentumCheckBox->Name = L"UseMomentumCheckBox";
            this->UseMomentumCheckBox->Size = System::Drawing::Size(112, 17);
            this->UseMomentumCheckBox->TabIndex = 10;
            this->UseMomentumCheckBox->Text = L"Use Momentum";
            this->UseMomentumCheckBox->UseVisualStyleBackColor = true;
            this->UseMomentumCheckBox->CheckedChanged += gcnew System::EventHandler(this, &Form1::UseMomentumCheckBox_CheckedChanged);
            // 
            // labelMomentum
            // 
            this->labelMomentum->AutoSize = true;
            this->labelMomentum->Location = System::Drawing::Point(107, 415);
            this->labelMomentum->Name = L"labelMomentum";
            this->labelMomentum->Size = System::Drawing::Size(67, 13);
            this->labelMomentum->TabIndex = 11;
            this->labelMomentum->Text = L"Momentum";
            // 
            // MomentumTextBox
            // 
            this->MomentumTextBox->Location = System::Drawing::Point(10, 408);
            this->MomentumTextBox->Name = L"MomentumTextBox";
            this->MomentumTextBox->Size = System::Drawing::Size(82, 20);
            this->MomentumTextBox->TabIndex = 12;
            this->MomentumTextBox->Text = L"0,9";
            // 
            // labelLearningRate
            // 
            this->labelLearningRate->AutoSize = true;
            this->labelLearningRate->Location = System::Drawing::Point(99, 289);
            this->labelLearningRate->Name = L"labelLearningRate";
            this->labelLearningRate->Size = System::Drawing::Size(87, 13);
            this->labelLearningRate->TabIndex = 8;
            this->labelLearningRate->Text = L"Learning Rate";
            // 
            // LearningRateTextBox
            // 
            this->LearningRateTextBox->Location = System::Drawing::Point(5, 286);
            this->LearningRateTextBox->Name = L"LearningRateTextBox";
            this->LearningRateTextBox->Size = System::Drawing::Size(82, 20);
            this->LearningRateTextBox->TabIndex = 9;
            this->LearningRateTextBox->Text = L"0,01";
            // 
            // Set_Net
            // 
            this->Set_Net->Location = System::Drawing::Point(6, 433);
            this->Set_Net->Name = L"Set_Net";
            this->Set_Net->Size = System::Drawing::Size(180, 45);
            this->Set_Net->TabIndex = 13;
            this->Set_Net->Text = L"Network Setting";
            this->Set_Net->UseVisualStyleBackColor = true;
            this->Set_Net->Click += gcnew System::EventHandler(this, &Form1::Set_Net_Click);
            // 
            // RegressionCheckBox
            // 
            this->RegressionCheckBox->AutoSize = true;
            this->RegressionCheckBox->Location = System::Drawing::Point(6, 263);
            this->RegressionCheckBox->Name = L"RegressionCheckBox";
            this->RegressionCheckBox->Size = System::Drawing::Size(124, 17);
            this->RegressionCheckBox->TabIndex = 14;
            this->RegressionCheckBox->Text = L"Regression Mode";
            this->RegressionCheckBox->UseVisualStyleBackColor = true;
            this->RegressionCheckBox->CheckedChanged += gcnew System::EventHandler(this, &Form1::RegressionCheckBox_CheckedChanged);
            // 
            // groupBox2
            // 
            this->groupBox2->BackColor = System::Drawing::SystemColors::ButtonHighlight;
            this->groupBox2->Controls->Add(this->btnTum);
            this->groupBox2->Controls->Add(this->btnSon);
            this->groupBox2->Controls->Add(this->label2);
            this->groupBox2->Controls->Add(this->ClassNoBox);
            this->groupBox2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold));
            this->groupBox2->Location = System::Drawing::Point(869, 540);
            this->groupBox2->Name = L"groupBox2";
            this->groupBox2->Size = System::Drawing::Size(200, 103);
            this->groupBox2->TabIndex = 2;
            this->groupBox2->TabStop = false;
            this->groupBox2->Text = L"Data Collection";
            // 
            // btnTum
            // 
            this->btnTum->Location = System::Drawing::Point(110, 69);
            this->btnTum->Name = L"btnTum";
            this->btnTum->Size = System::Drawing::Size(75, 23);
            this->btnTum->TabIndex = 3;
            this->btnTum->Text = L"TÜM";
            this->btnTum->UseVisualStyleBackColor = true;
            this->btnTum->Click += gcnew System::EventHandler(this, &Form1::btnTum_Click);
            // 
            // btnSon
            // 
            this->btnSon->Location = System::Drawing::Point(10, 69);
            this->btnSon->Name = L"btnSon";
            this->btnSon->Size = System::Drawing::Size(75, 23);
            this->btnSon->TabIndex = 2;
            this->btnSon->Text = L"SON";
            this->btnSon->UseVisualStyleBackColor = true;
            this->btnSon->Click += gcnew System::EventHandler(this, &Form1::btnSon_Click);
            // 
            // label2
            // 
            this->label2->AutoSize = true;
            this->label2->Location = System::Drawing::Point(98, 28);
            this->label2->Name = L"label2";
            this->label2->Size = System::Drawing::Size(81, 13);
            this->label2->TabIndex = 1;
            this->label2->Text = L"Örnek Etiketi";
            // 
            // ClassNoBox
            // 
            this->ClassNoBox->FormattingEnabled = true;
            this->ClassNoBox->Items->AddRange(gcnew cli::array< System::Object^  >(9) {
                L"1", L"2", L"3", L"4", L"5", L"6", L"7", L"8",
                    L"9"
            });
            this->ClassNoBox->Location = System::Drawing::Point(10, 25);
            this->ClassNoBox->Name = L"ClassNoBox";
            this->ClassNoBox->Size = System::Drawing::Size(82, 21);
            this->ClassNoBox->TabIndex = 0;
            this->ClassNoBox->Text = L"1";
            // 
            // label3
            // 
            this->label3->AutoSize = true;
            this->label3->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold));
            this->label3->Location = System::Drawing::Point(866, 646);
            this->label3->Name = L"label3";
            this->label3->Size = System::Drawing::Size(120, 15);
            this->label3->TabIndex = 3;
            this->label3->Text = L"Samples Count: 0";
            // 
            // menuStrip1
            // 
            this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
                this->fileToolStripMenuItem,
                    this->processToolStripMenuItem
            });
            this->menuStrip1->Location = System::Drawing::Point(0, 0);
            this->menuStrip1->Name = L"menuStrip1";
            this->menuStrip1->Size = System::Drawing::Size(1450, 24);
            this->menuStrip1->TabIndex = 4;
            this->menuStrip1->Text = L"menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
                this->readDataToolStripMenuItem,
                    this->saveDataToolStripMenuItem
            });
            this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
            this->fileToolStripMenuItem->Size = System::Drawing::Size(37, 20);
            this->fileToolStripMenuItem->Text = L"File";
            // 
            // readDataToolStripMenuItem
            // 
            this->readDataToolStripMenuItem->Name = L"readDataToolStripMenuItem";
            this->readDataToolStripMenuItem->Size = System::Drawing::Size(129, 22);
            this->readDataToolStripMenuItem->Text = L"Read_Data";
            this->readDataToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::readDataToolStripMenuItem_Click);
            // 
            // saveDataToolStripMenuItem
            // 
            this->saveDataToolStripMenuItem->Name = L"saveDataToolStripMenuItem";
            this->saveDataToolStripMenuItem->Size = System::Drawing::Size(129, 22);
            this->saveDataToolStripMenuItem->Text = L"Save_Data";
            this->saveDataToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::saveDataToolStripMenuItem_Click);
            // 
            // processToolStripMenuItem
            // 
            this->processToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
                this->trainingToolStripMenuItem,
                    this->testingToolStripMenuItem
            });
            this->processToolStripMenuItem->Name = L"processToolStripMenuItem";
            this->processToolStripMenuItem->Size = System::Drawing::Size(59, 20);
            this->processToolStripMenuItem->Text = L"Process";
            // 
            // trainingToolStripMenuItem
            // 
            this->trainingToolStripMenuItem->Name = L"trainingToolStripMenuItem";
            this->trainingToolStripMenuItem->Size = System::Drawing::Size(117, 22);
            this->trainingToolStripMenuItem->Text = L"Training";
            this->trainingToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::trainingToolStripMenuItem_Click);
            // 
            // testingToolStripMenuItem
            // 
            this->testingToolStripMenuItem->Name = L"testingToolStripMenuItem";
            this->testingToolStripMenuItem->Size = System::Drawing::Size(117, 22);
            this->testingToolStripMenuItem->Text = L"Testing";
            this->testingToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::testingToolStripMenuItem_Click);
            // 
            // chartEpoch
            // 
            chartArea1->Name = L"ChartArea1";
            this->chartEpoch->ChartAreas->Add(chartArea1);
            this->chartEpoch->Location = System::Drawing::Point(1075, 340);
            this->chartEpoch->Name = L"chartEpoch";
            series1->BorderWidth = 2;
            series1->ChartArea = L"ChartArea1";
            series1->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Spline;
            series1->Color = System::Drawing::Color::Red;
            series1->Name = L"Error";
            this->chartEpoch->Series->Add(series1);
            this->chartEpoch->Size = System::Drawing::Size(360, 290);
            this->chartEpoch->TabIndex = 10;
            this->chartEpoch->Text = L"Epoch Error Graph";
            title1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold));
            title1->Name = L"Title1";
            title1->Text = L"Training Error";
            this->chartEpoch->Titles->Add(title1);
            // 
            // openFileDialog1
            // 
            this->openFileDialog1->FileName = L"openFileDialog1";
            // 
            // textBox1
            // 
            this->textBox1->Location = System::Drawing::Point(1075, 50);
            this->textBox1->Multiline = true;
            this->textBox1->Name = L"textBox1";
            this->textBox1->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
            this->textBox1->Size = System::Drawing::Size(360, 280);
            this->textBox1->TabIndex = 5;
            // 
            // Form1
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
            this->BackColor = System::Drawing::SystemColors::ActiveCaption;
            this->ClientSize = System::Drawing::Size(1450, 670);
            this->Controls->Add(this->chartEpoch);
            this->Controls->Add(this->textBox1);
            this->Controls->Add(this->label3);
            this->Controls->Add(this->groupBox2);
            this->Controls->Add(this->groupBox1);
            this->Controls->Add(this->pictureBox1);
            this->Controls->Add(this->menuStrip1);
            this->MainMenuStrip = this->menuStrip1;
            this->Name = L"Form1";
            this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
            this->Text = L"Neural Network Trainer - Dynamic Multi-Layer";
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
            this->groupBox1->ResumeLayout(false);
            this->groupBox1->PerformLayout();
            this->groupBox2->ResumeLayout(false);
            this->groupBox2->PerformLayout();
            this->menuStrip1->ResumeLayout(false);
            this->menuStrip1->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chartEpoch))->EndInit();
            this->ResumeLayout(false);
            this->PerformLayout();

        }



        void draw_current_decision_lines(float* current_weights, float* current_bias,
            float* mean, float* std, int epoch, float error, bool isFinal) {
            isFinal = false;
            Bitmap^ bmp = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
            Graphics^ g = Graphics::FromImage(bmp);
            g->Clear(Color::White);

            int cx = pictureBox1->Width / 2;
            int cy = pictureBox1->Height / 2;

            Pen^ axisPen = gcnew Pen(Color::LightGray, 2);
            g->DrawLine(axisPen, cx, 0, cx, pictureBox1->Height);
            g->DrawLine(axisPen, 0, cy, pictureBox1->Width, cy);

            array<Color>^ lineColors = gcnew array<Color>(7) {
                    Color::Black,   // Class 0
                    Color::Red,     // Class 1
                    Color::Blue,    // Class 2
                    Color::Green,   // Class 3
                    Color::Yellow,  // Class 4
                    Color::Orange,  // Class 5
                    Color::Purple   // Class 6
            };

            int numOutNeuron = (class_count > 2) ? class_count : 1;

            for (int k = 0; k < numOutNeuron; k++) {
                float w1, w2, b;

                if (numOutNeuron == 1) {
                    w1 = current_weights[0];
                    w2 = current_weights[1];
                    b = current_bias[0];
                }
                else {
                    w1 = current_weights[k * inputDim + 0];
                    w2 = current_weights[k * inputDim + 1];
                    b = current_bias[k];
                }

                if (fabs(w2) < 1e-6) continue;

                float x_left = -cx;
                float x_right = cx;

                float x_norm_left = (x_left - mean[0]) / std[0];
                float x_norm_right = (x_right - mean[0]) / std[0];

                float y_norm_left = -(w1 * x_norm_left + b) / w2;
                float y_norm_right = -(w1 * x_norm_right + b) / w2;

                float y_left = y_norm_left * std[1] + mean[1];
                float y_right = y_norm_right * std[1] + mean[1];

                int sx1 = (int)(x_left + cx);
                int sy1 = (int)(cy - y_left);
                int sx2 = (int)(x_right + cx);
                int sy2 = (int)(cy - y_right);

                Pen^ dashedPen = gcnew Pen(lineColors[k % 7], isFinal ? 4 : 3);

                if (!isFinal) {
                    dashedPen->DashStyle = System::Drawing::Drawing2D::DashStyle::Dash;
                    dashedPen->DashPattern = gcnew array<float>{10, 5}; 
                }

                Pen^ shadowPen = gcnew Pen(Color::FromArgb(30, 0, 0, 0), isFinal ? 6 : 5);
                if (!isFinal) {
                    shadowPen->DashStyle = System::Drawing::Drawing2D::DashStyle::Dash;
                }
                g->DrawLine(shadowPen, sx1 + 2, sy1 + 2, sx2 + 2, sy2 + 2);

                g->DrawLine(dashedPen, sx1, sy1, sx2, sy2);

                if (numOutNeuron == 1) break;
            }

            array<Color>^ sampleColors = gcnew array<Color>(7) {
                Color::Black, Color::Red, Color::Blue, Color::Green,
                    Color::Yellow, Color::Orange, Color::Purple
            };

            for (int i = 0; i < numSample; i++) {
                int px = (int)(Samples[i * inputDim + 0] + cx);
                int py = (int)(cy - Samples[i * inputDim + 1]);

                Pen^ pen = gcnew Pen(sampleColors[(int)targets[i] % 7], 4);
                Brush^ brush = gcnew SolidBrush(sampleColors[(int)targets[i] % 7]);

                g->DrawLine(pen, px - 7, py, px + 7, py);
                g->DrawLine(pen, px, py - 7, px, py + 7);
                g->FillEllipse(brush, px - 3, py - 3, 6, 6);
            }

            String^ info = isFinal
                ? "✓ EĞİTİM TAMAMLANDI | Final Epoch: " + epoch + " | Final Error: " + error.ToString("F4")
                : "Epoch: " + epoch + " | Error: " + error.ToString("F4");

            System::Drawing::Font^ font = gcnew System::Drawing::Font("Arial", isFinal ? 14 : 12,
                isFinal ? FontStyle::Bold : FontStyle::Bold);
            Brush^ textBrush = gcnew SolidBrush(isFinal ? Color::DarkGreen : Color::Black);
            g->DrawString(info, font, textBrush, 10, 10);

            pictureBox1->Image = bmp;
            pictureBox1->Refresh();
        }

        void draw_current_regression_curve(int epoch, float err, bool isFinal)
        {
            if (numSample <= 0 || Samples == nullptr || targets == nullptr) {
                return;
            }

            Bitmap^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
            Graphics^ g = Graphics::FromImage(surface);
            g->Clear(Color::White);
            g->SmoothingMode = System::Drawing::Drawing2D::SmoothingMode::AntiAlias;

            int cx = pictureBox1->Width / 2;
            int cy = pictureBox1->Height / 2;

            Pen^ axisPen = gcnew Pen(Color::LightGray, 2);
            g->DrawLine(axisPen, cx, 0, cx, pictureBox1->Height);
            g->DrawLine(axisPen, 0, cy, pictureBox1->Width, cy);

            int step = isFinal ? 2 : 10;  

            Pen^ curvePen = gcnew Pen(isFinal ? Color::DarkRed : Color::Red, isFinal ? 4 : 2);

            float* input1 = new float[inputDim];
            float* input2 = new float[inputDim];

            for (int column = 0; column < pictureBox1->Width - step; column += step)
            {
                float x1 = float(column - cx);
                float x2 = float(column + step - cx);

                input1[0] = x1;
                input2[0] = x2;

                for (int d = 1; d < inputDim; ++d) {
                    input1[d] = 0.0f;
                    input2[d] = 0.0f;
                }

                float y1, y2;

                if (useDynamicLayers && dynamicRegressionWeights != nullptr) {
                    y1 = EvalMLRegression(
                        input1, dynamicRegressionWeights,
                        mean_params, std_params, mean_y, std_y
                    );
                    y2 = EvalMLRegression(
                        input2, dynamicRegressionWeights,
                        mean_params, std_params, mean_y, std_y
                    );
                }
                else if (regressionWeights != nullptr) {
                    RegressionConfig cfg;
                    cfg.inputDim = inputDim;
                    cfg.outputDim = 1;

                    y1 = EvalSLRegression(
                        input1, regressionWeights, cfg,
                        mean_params, std_params, mean_y, std_y
                    );
                    y2 = EvalSLRegression(
                        input2, regressionWeights, cfg,
                        mean_params, std_params, mean_y, std_y
                    );
                }
                else {
                    delete[] input1;
                    delete[] input2;
                    break;
                }

                if (System::Single::IsNaN(y1) || System::Single::IsInfinity(y1) ||
                    System::Single::IsNaN(y2) || System::Single::IsInfinity(y2)) {
                    continue;
                }

                if (y1 > 10000) y1 = 10000;
                if (y1 < -10000) y1 = -10000;
                if (y2 > 10000) y2 = 10000;
                if (y2 < -10000) y2 = -10000;

                int px1 = column;
                int py1 = cy - (int)y1;
                int px2 = column + step;
                int py2 = cy - (int)y2;

                if (py1 > -5000 && py1 < 5000 && py2 > -5000 && py2 < 5000) {
                    g->DrawLine(curvePen, px1, py1, px2, py2);
                }
            }

            delete[] input1;
            delete[] input2;

            try {
                for (int i = 0; i < numSample; i++) {
                    if (Samples == nullptr || targets == nullptr) break;

                    int sx = (int)Samples[i * inputDim] + cx;
                    int sy = cy - (int)targets[i];

                    g->FillEllipse(gcnew SolidBrush(Color::Blue), sx - 5, sy - 5, 10, 10);
                    g->DrawEllipse(gcnew Pen(Color::DarkBlue, 2), sx - 5, sy - 5, 10, 10);
                }
            }
            catch (Exception^ ex) {
                System::Diagnostics::Debug::WriteLine("draw_current_regression_curve error: " + ex->Message);
            }

            String^ info;
            if (isFinal) {
                info = "✓ REGRESYON EĞİTİMİ BİTTİ | Epoch: " + epoch + " | Error: " + err.ToString("F6");
            }
            else {
                info = "Epoch: " + epoch + " | Error: " + err.ToString("F6");
            }

            System::Drawing::Font^ font = gcnew System::Drawing::Font("Arial", 12,
                isFinal ? FontStyle::Bold : FontStyle::Regular);
            Brush^ textBrush = gcnew SolidBrush(isFinal ? Color::DarkGreen : Color::Black);

            Brush^ shadowBrush = gcnew SolidBrush(Color::FromArgb(200, 255, 255, 255));
            g->FillRectangle(shadowBrush, 5, 5, 450, 30);

            g->DrawString(info, font, textBrush, 10, 10);

            pictureBox1->Image = surface;
            pictureBox1->Refresh();
        }

           void chart_add_point(int epoch, float err)
           {
               chartEpoch->Series["Error"]->Points->AddXY(epoch, err);
           }
#pragma region Draw Sample
           void draw_sample(int temp_x, int temp_y, int label) {
               Pen^ pen;
               switch (label) {
               case 0: pen = gcnew Pen(Color::Black, 3.0f); break;
               case 1: pen = gcnew Pen(Color::Red, 3.0f); break;
               case 2: pen = gcnew Pen(Color::Blue, 3.0f); break;
               case 3: pen = gcnew Pen(Color::Green, 3.0f); break;
               case 4: pen = gcnew Pen(Color::Yellow, 3.0f); break;
               case 5: pen = gcnew Pen(Color::Orange, 3.0f); break;
               default: pen = gcnew Pen(Color::YellowGreen, 3.0f);
               }
               pictureBox1->CreateGraphics()->DrawLine(pen, temp_x - 5, temp_y, temp_x + 5, temp_y);
               pictureBox1->CreateGraphics()->DrawLine(pen, temp_x, temp_y - 5, temp_x, temp_y + 5);
           }

           void Redraw_All_Samples() {
               Bitmap^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
               Graphics^ g = Graphics::FromImage(surface);
               g->Clear(Color::White);

               Pen^ axisPen = gcnew Pen(Color::Black, 3.0f);
               int cx = pictureBox1->Width / 2;
               int cy = pictureBox1->Height / 2;
               g->DrawLine(axisPen, cx, 0, cx, pictureBox1->Height);
               g->DrawLine(axisPen, 0, cy, pictureBox1->Width, cy);

               for (int i = 0; i < numSample; i++) {
                   int px, py;

                   if (isRegression) {
                       float x_val = Samples[i * inputDim + 0];
                       float y_val = targets[i];

                       px = (int)(x_val + cx);
                       py = (int)(cy - y_val);

                       g->FillEllipse(gcnew SolidBrush(Color::Blue), px - 3, py - 3, 6, 6);
                   }
                   else {
                       px = (int)(Samples[i * inputDim] + cx);
                       py = (int)(cy - Samples[i * inputDim + 1]);

                       int label = (int)targets[i];

                       Pen^ pen;
                       switch (label) {
                       case 0: pen = gcnew Pen(Color::Black, 3.0f); break;
                       case 1: pen = gcnew Pen(Color::Red, 3.0f); break;
                       case 2: pen = gcnew Pen(Color::Blue, 3.0f); break;
                       case 3: pen = gcnew Pen(Color::Green, 3.0f); break;
                       case 4: pen = gcnew Pen(Color::Yellow, 3.0f); break;
                       case 5: pen = gcnew Pen(Color::Orange, 3.0f); break;
                       default: pen = gcnew Pen(Color::YellowGreen, 3.0f);
                       }

                       g->DrawLine(pen, px - 5, py, px + 5, py);
                       g->DrawLine(pen, px, py - 5, px, py + 5);
                   }
               }

               pictureBox1->Image = surface;
               pictureBox1->Refresh();
               label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
           }

    private: System::Void btnTum_Click(System::Object^ sender, System::EventArgs^ e) {
        if (numSample == 0) return;

        if (Samples != nullptr) {
            delete[] Samples;
            Samples = nullptr;
        }
        if (targets != nullptr) {
            delete[] targets;
            targets = nullptr;
        }

        numSample = 0;

        Redraw_All_Samples();

        chartEpoch->Series["Error"]->Points->Clear();
        textBox1->Text = "";
    }

    private: System::Void btnSon_Click(System::Object^ sender, System::EventArgs^ e) {
        if (numSample <= 0) {
            MessageBox::Show("Silinecek veri yok!");
            return;
        }

        if (numSample == 1) {
            btnTum_Click(sender, e);
            return;
        }

        int newCount = numSample - 1;
        float* newSamples = new float[newCount * inputDim];
        float* newTargets = new float[newCount];

        for (int i = 0; i < newCount; i++) {
            newTargets[i] = targets[i];

            for (int j = 0; j < inputDim; j++) {
                newSamples[i * inputDim + j] = Samples[i * inputDim + j];
            }
        }

        delete[] Samples;
        delete[] targets;

        Samples = newSamples;
        targets = newTargets;

        numSample = newCount;

        Redraw_All_Samples();
    }

    private: System::Void DynamicLayersCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
        useDynamicLayers = DynamicLayersCheckBox->Checked;

        if (useDynamicLayers) {
            labelNumLayers->Visible = true;
            NumLayersBox->Visible = true;

            UpdateLayerNeuronsPanel();
        }
        else {
            labelNumLayers->Visible = false;
            NumLayersBox->Visible = false;
            layerNeuronsPanel->Visible = false;
        }
    }

    private: System::Void NumLayersBox_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
        numHiddenLayers = Convert::ToInt32(NumLayersBox->Text);
        UpdateLayerNeuronsPanel();
    }

    private: System::Void RegressionCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
        isRegression = RegressionCheckBox->Checked;

        if (isRegression) {
            ClassCountBox->Enabled = false;
            ClassNoBox->Enabled = false;
            label1->Enabled = false;
            label2->Enabled = false;

            inputDim = 1;  

            MessageBox::Show(
                "Regresyon Modu Aktif!\n\n"
                "• Mouse tıkladığınızda X koordinatı girdi,\n"
                "  Y koordinatı hedef değer olarak kaydedilir.\n\n"
                "• Eğitim sonrası test ederken, tıkladığınız\n"
                "  X koordinatına göre tahmin edilen Y değeri\n"
                "  bir çizgi olarak gösterilir.",
                "Regresyon Modu"
            );
        }
        else {
            ClassCountBox->Enabled = true;
            ClassNoBox->Enabled = true;
            label1->Enabled = true;
            label2->Enabled = true;

            inputDim = 2;  
        }
    }

    private: void UpdateLayerNeuronsPanel() {
        // Paneli temizle
        layerNeuronsPanel->Controls->Clear();
        layerNeurons->clear();

        if (!useDynamicLayers) return;

        layerNeuronsPanel->Visible = true;

        int yPos = 5;

        for (int i = 0; i < numHiddenLayers; i++) {
            // Label oluştur
            Label^ lbl = gcnew Label();
            lbl->AutoSize = true;
            lbl->Location = System::Drawing::Point(5, yPos);
            lbl->Text = "Layer " + (i + 1) + ":";
            lbl->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8, System::Drawing::FontStyle::Bold));
            layerNeuronsPanel->Controls->Add(lbl);

            // ComboBox oluştur
            ComboBox^ cb = gcnew ComboBox();
            cb->Location = System::Drawing::Point(70, yPos - 2);
            cb->Size = System::Drawing::Size(100, 21);
            cb->Name = "Layer" + i;

            // Nöron sayısı seçenekleri
            cb->Items->AddRange(gcnew cli::array< System::Object^  >(12) {
                L"2", L"3", L"4", L"5", L"6", L"8", L"10", L"12", L"16", L"20", L"24", L"32"
            });
            cb->Text = (i == 0) ? L"8" : (i == numHiddenLayers - 1 ? L"4" : L"6");

            layerNeuronsPanel->Controls->Add(cb);

            yPos += 30;
        }
    }
#pragma endregion
    private: System::Void pictureBox1_MouseClick(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
        if (isRegression) {

            float* x = new float[inputDim];   
            int temp_x = System::Convert::ToInt32(e->X);
            int temp_y = System::Convert::ToInt32(e->Y);

            x[0] = float(temp_x - (pictureBox1->Width / 2));

            float y_target = float(pictureBox1->Height / 2 - temp_y);

            if (numSample == 0) {
                numSample = 1;
                Samples = new float[numSample * inputDim]; 
                targets = new float[numSample];

                Samples[0] = x[0];
                targets[0] = y_target;
            }
            else {
                int newCount = numSample + 1;

                float* newSamples = new float[newCount * inputDim];
                float* newTargets = new float[newCount];

                for (int i = 0; i < numSample; i++) {
                    for (int j = 0; j < inputDim; j++) {
                        newSamples[i * inputDim + j] = Samples[i * inputDim + j];
                    }
                    newTargets[i] = targets[i];
                }

                newSamples[(newCount - 1) * inputDim + 0] = x[0];
                newTargets[newCount - 1] = y_target;

                delete[] Samples;
                delete[] targets;

                Samples = newSamples;
                targets = newTargets;

                numSample = newCount;
            }

            pictureBox1->CreateGraphics()->FillEllipse(
                gcnew SolidBrush(Color::Blue),
                temp_x - 3, temp_y - 3, 6, 6
            );

            label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
            delete[] x;
        }
        else {
            if (class_count == 0)
                MessageBox::Show("The Network Architecture should be firstly set up");
            else {
                float* x = new float[inputDim];
                int temp_x = (System::Convert::ToInt32(e->X));
                int temp_y = (System::Convert::ToInt32(e->Y));
                x[0] = float(temp_x - (pictureBox1->Width / 2));
                x[1] = float(pictureBox1->Height / 2 - temp_y);
                int label;
                int numLabel = Convert::ToInt32(ClassNoBox->Text);
                if (numLabel > class_count)
                    MessageBox::Show("The class label cannot be greater than the maximum number of classes.");
                else {
                    label = numLabel - 1;
                    if (numSample == 0) {
                        numSample = 1;
                        Samples = new float[numSample * inputDim];
                        targets = new float[numSample];
                        for (int i = 0; i < inputDim; i++)
                            Samples[i] = x[i];
                        targets[0] = float(label);
                    }
                    else {
                        numSample++;
                        Samples = Add_Data(Samples, numSample, x, inputDim);
                        targets = Add_Labels(targets, numSample, label);
                    }
                    draw_sample(temp_x, temp_y, label);
                    label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
                    delete[] x;
                }
            }
        }
    }
    private: System::Void pictureBox1_Paint(System::Object^ sender, System::Windows::Forms::PaintEventArgs^ e) {
        Pen^ pen = gcnew Pen(Color::Black, 3.0f);
        int center_width, center_height;
        center_width = (int)(pictureBox1->Width / 2);
        center_height = (int)(pictureBox1->Height / 2);
        e->Graphics->DrawLine(pen, center_width, 0, center_width, pictureBox1->Height);
        e->Graphics->DrawLine(pen, 0, center_height, pictureBox1->Width, center_height);
    }
    private: System::Void Set_Net_Click(System::Object^ sender, System::EventArgs^ e) {
        class_count = Convert::ToInt32(ClassCountBox->Text);

        if (isRegression) {
            inputDim = 1;
            if (useDynamicLayers) {
                layerNeurons->clear();
                for each (Control ^ ctrl in layerNeuronsPanel->Controls) {
                    if (ctrl->GetType() == ComboBox::typeid) {
                        ComboBox^ cb = safe_cast<ComboBox^>(ctrl);
                        layerNeurons->push_back(Convert::ToInt32(cb->Text));
                    }
                }

                if (layerNeurons->size() != numHiddenLayers) {
                    MessageBox::Show("Tüm katmanlar için nöron sayısı seçilmeli!");
                    return;
                }

                if (dynamicRegressionWeights != nullptr) {
                    delete dynamicRegressionWeights;
                }

                dynamicRegressionWeights = new DynamicRegressionWeights(
                    inputDim, *layerNeurons, 1);  

                String^ msg = "Regresyon Ağ Mimarisi:\n";
                msg += "Giriş: " + inputDim + " nöron\n";
                for (int i = 0; i < numHiddenLayers; i++) {
                    msg += "Gizli Katman " + (i + 1) + ": " + (*layerNeurons)[i] + " nöron\n";
                }
                msg += "Çıkış: 1 nöron (Regresyon)\n";
                msg += "Learning Rate: " + LearningRateTextBox->Text + "\n";
                msg += "Momentum: " + (useMomentum ? MomentumTextBox->Text : "Kapalı");

                MessageBox::Show(msg, "Multi-Layer Regresyon Hazır");
            }
            else {
                if (regressionWeights == nullptr) {
                    regressionWeights = new RegressionWeights();
                }

                regressionWeights->W_output = init_array_random(inputDim);
                regressionWeights->b_output = init_array_random(1);

                MessageBox::Show(
                    "Single Layer Regresyon Ağı Hazır!\n\n"
                    "Giriş: 2D (X, Y koordinat)\n"
                    "Çıkış: 1D (Tahmin değeri)\n"
                    "Learning Rate: " + LearningRateTextBox->Text + "\n"
                    "Momentum: " + (useMomentum ? MomentumTextBox->Text : "Kapalı"),
                    "Regresyon Ağı Hazır"
                );
            }

            Set_Net->Text = "Regression Ready!";
        }
        else {
            inputDim = 2;
            class_count = Convert::ToInt32(ClassCountBox->Text);

            if (useMomentum) {
                try {
                    momentumValue = Convert::ToSingle(MomentumTextBox->Text);
                    if (momentumValue < 0.0f || momentumValue > 1.0f) {
                        MessageBox::Show("Momentum değeri 0 ile 1 arasında olmalıdır!");
                        return;
                    }
                }
                catch (Exception^) {
                    MessageBox::Show("Geçersiz momentum değeri!");
                    return;
                }
            }
            else {
                momentumValue = 0.0f;
            }

            if (useDynamicLayers) {
                layerNeurons->clear();
                for each (Control ^ ctrl in layerNeuronsPanel->Controls) {
                    if (ctrl->GetType() == ComboBox::typeid) {
                        ComboBox^ cb = safe_cast<ComboBox^>(ctrl);
                        layerNeurons->push_back(Convert::ToInt32(cb->Text));
                    }
                }

                if (layerNeurons->size() != numHiddenLayers) {
                    MessageBox::Show("Tüm katmanlar için nöron sayısı seçilmeli!");
                    return;
                }

                if (dynamicWeights != nullptr) {
                    delete dynamicWeights;
                }

                int outputDim = (class_count > 2) ? class_count : 1;
                dynamicWeights = new DynamicMultiLayerWeights(inputDim, *(layerNeurons), outputDim);

                String^ msg = "Ağ Mimarisi:\n";
                msg += "Giriş: " + inputDim + " nöron\n";
                for (int i = 0; i < numHiddenLayers; i++) {
                    msg += "Gizli Katman " + (i + 1) + ": " + (*layerNeurons)[i] + " nöron\n";
                }
                msg += "Çıkış: " + outputDim + " nöron\n";
                msg += "Momentum: " + (useMomentum ? momentumValue.ToString("F2") : "Kapalı");

                Set_Net->Text = "Network Ready!";
                MessageBox::Show(msg, "Ağ Hazır");
            }
            else {
                Weights = new float[class_count * inputDim];
                bias = new float[class_count];

                if (class_count > 2) {
                    Weights = init_array_random(class_count * inputDim);
                    bias = init_array_random(class_count);
                }
                else {
                    int numOutNeuron = 1;
                    Weights = init_array_random(inputDim);
                    bias = init_array_random(numOutNeuron);
                }
                Set_Net->Text = "Single Layer Ready";
            }
        }
    }
    private: System::Void readDataToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
        char** c = new char* [2];
        MessageBox::Show("Veri Kümesini Yükleyin");
        c[0] = "../Data/Samples.txt";
        c[1] = "../Data/weights.txt";
        std::ifstream file;
        int num, w, h, Dim, label;
        file.open(c[0]);
        if (file.is_open()) {
            file >> Dim >> w >> h >> num;
            textBox1->Text += "Dimension: " + Convert::ToString(Dim) + "- Width: " + Convert::ToString(w) + " - Height: " + Convert::ToString(h) + " - Number of Class: " + Convert::ToString(num) + "\r\n";
            class_count = num;
            inputDim = Dim;
            Weights = new float[class_count * inputDim];
            bias = new float[class_count];
            numSample = 0;
            float* x = new float[inputDim];
            while (!file.eof())
            {
                if (numSample == 0) { 
                    numSample = 1;
                    Samples = new float[inputDim]; targets = new float[numSample];
                    for (int i = 0; i < inputDim; i++)
                        file >> Samples[i];
                    file >> targets[0];
                }
                else {

                    for (int i = 0; i < inputDim; i++)
                        file >> x[i];
                    file >> label;
                    if (!file.eof()) {
                        numSample++;
                        Samples = Add_Data(Samples, numSample, x, inputDim);
                        targets = Add_Labels(targets, numSample, label);
                    }
                }
            } 
            delete[]x;
            file.close();
            for (int i = 0; i < numSample; i++) {
                draw_sample(Samples[i * inputDim] + w, h - Samples[i * inputDim + 1], targets[i]);
                for (int j = 0; j < inputDim; j++)
                    textBox1->Text += Convert::ToString(Samples[i * inputDim + j]) + " ";
                textBox1->Text += Convert::ToString(targets[i]) + "\r\n";
            }
            label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
            MessageBox::Show("Dosya basari ile okundu");
        }
        else MessageBox::Show("Dosya acilamadi");
        int Layer;
        file.open(c[1]);
        if (file.is_open()) {
            file >> Layer >> Dim >> num;
            class_count = num;
            inputDim = Dim;
            Weights = new float[class_count * inputDim];
            bias = new float[class_count];
            textBox1->Text += "Layer: " + Convert::ToString(Layer) + " Dimension: " + Convert::ToString(Dim) + " numClass:" + Convert::ToString(num) + "\r\n";
            while (!file.eof())
            {
                for (int i = 0; i < class_count; i++)
                    for (int j = 0; j < inputDim; j++)
                        file >> Weights[i * inputDim + j];
                for (int i = 0; i < class_count; i++)
                    file >> bias[i];
            }
            file.close();
        }
        delete[]c;
    }
    private: System::Void saveDataToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
        if (numSample != 0) {
            char** c = new char* [2];
            c[0] = "../Data/Samples.txt";
            c[1] = "../Data/weights.txt";
            std::ofstream ofs(c[0]);
            if (!ofs.bad()) {
                ofs << inputDim << " " << pictureBox1->Width / 2 << " " << pictureBox1->Height / 2 << " " << class_count << std::endl;
                for (int i = 0; i < numSample; i++) {
                    for (int d = 0; d < inputDim; d++)
                        ofs << Samples[i * inputDim + d] << " ";
                    ofs << targets[i] << std::endl;
                }
                ofs.close();
            }
            else MessageBox::Show("Samples icin dosya acilamadi");
            std::ofstream file(c[1]);
            if (!file.bad()) {
                file << 1 << " " << inputDim << " " << class_count << std::endl;
                for (int k = 0; k < class_count * inputDim; k++)
                    file << Weights[k] << " ";
                file << std::endl;
                for (int k = 0; k < class_count; k++)
                    file << bias[k] << " ";
                file.close();
            }
            else MessageBox::Show("Weight icin dosya acilamadi");
            delete[]c;
        }
        else MessageBox::Show("At least one sample should be given");
    }

    private: System::Void trainingToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
        if (Samples == nullptr || targets == nullptr || numSample <= 0)
        {
            MessageBox::Show("Eğitim verisi yok.");
            return;
        }
        
        chartEpoch->Series["Error"]->Points->Clear();
        textBox1->Text = "";

        if (mean_params != nullptr) delete[] mean_params;
        if (std_params != nullptr) delete[] std_params;
        mean_params = new float[inputDim];
        std_params = new float[inputDim];
        Z_Score_Parameters(Samples, numSample, inputDim, mean_params, std_params);

        float Lrate = 0.01f;
        int maxEpoch = 2000;
        int patience = 100;
        float improvementThreshold = 0.001f;
        float momentum = useMomentum ? momentumValue : 0.0f;

        try {
            Lrate = Convert::ToSingle(LearningRateTextBox->Text);
            maxEpoch = Convert::ToInt32(MaxEpochTextBox->Text);
            patience = Convert::ToInt32(PatienceTextBox->Text);
            improvementThreshold = Convert::ToSingle(EarlyStopTextBox->Text);
        }
        catch (Exception^) {
            MessageBox::Show("Parametre hatası! Varsayılanlar kullanılacak.");
        }

        float* errorHistory = new float[maxEpoch];
        int finalEpoch = 0;

        int visualizationInterval = (maxEpoch <= 1000) ? 20 : 50;

        if (isRegression) {
            mean_y = 0.0f;
            for (int i = 0; i < numSample; i++) mean_y += targets[i];
            mean_y /= numSample;

            std_y = 0.0f;
            for (int i = 0; i < numSample; i++)
                std_y += (targets[i] - mean_y) * (targets[i] - mean_y);
            std_y = std::sqrt(std_y / numSample);
            if (std_y < 1e-6f) std_y = 1.0f;

            textBox1->Text += "=== REGRESYON EĞİTİMİ BAŞLADI ===\r\n";

            if (useDynamicLayers && dynamicRegressionWeights != nullptr) {
                finalEpoch = TrainMLRegression(
                    Samples, targets, numSample, dynamicRegressionWeights,
                    Lrate, maxEpoch, mean_params, std_params, mean_y, std_y,
                    errorHistory, momentum, patience, improvementThreshold
                );
            }
            else if (regressionWeights != nullptr) {
                RegressionConfig config;
                config.inputDim = inputDim;
                config.outputDim = 1;
                finalEpoch = TrainSLRegression(
                    Samples, targets, numSample, config, regressionWeights,
                    Lrate, maxEpoch, mean_params, std_params, mean_y, std_y,
                    errorHistory, momentum, patience, improvementThreshold
                );
            }

            for (int epoch = 0; epoch < finalEpoch; epoch++) {
                if (epoch % visualizationInterval == 0 || epoch == finalEpoch - 1) {
                    chart_add_point(epoch, errorHistory[epoch]);
                    if (epoch % (visualizationInterval * 2) == 0 || epoch == finalEpoch - 1) {
                        draw_current_regression_curve(epoch, errorHistory[epoch], (epoch == finalEpoch - 1));
                    }
                    Application::DoEvents();
                }
            }
        }
        else {
            textBox1->Text += "=== SINIFLANDIRMA EĞİTİMİ BAŞLADI ===\r\n";

            if (useDynamicLayers && dynamicWeights != nullptr) {
                DynamicMultiLayerConfig config;
                config.inputDim = inputDim;
                config.hiddenLayers = *(layerNeurons);
                config.outputDim = (class_count > 2) ? class_count : 1;

                finalEpoch = TrainMLClassify(
                    Samples, targets, numSample, config, dynamicWeights,
                    Lrate, maxEpoch, mean_params, std_params,
                    errorHistory, momentum, patience, improvementThreshold
                );

                for (int epoch = 0; epoch < finalEpoch; epoch++) {
                    if (epoch % visualizationInterval == 0 || epoch == finalEpoch - 1) {
                        chart_add_point(epoch, errorHistory[epoch]);
                        Application::DoEvents();
                    }
                }
            }
            else if (Weights != nullptr && bias != nullptr) {
                NetworkConfig config;
                config.inputDim = inputDim;
                config.outputDim = (class_count > 2) ? class_count : 1;
                NetworkWeights singleWeights;
                singleWeights.W_output = Weights;
                singleWeights.b_output = bias;

                finalEpoch = TrainSLClassify(
                    Samples, targets, numSample, config, &singleWeights,
                    Lrate, maxEpoch, mean_params, std_params,
                    errorHistory, momentum, patience, improvementThreshold
                );

                for (int epoch = 0; epoch < finalEpoch; epoch++) {
                    if (epoch % visualizationInterval == 0 || epoch == finalEpoch - 1) {
                        chart_add_point(epoch, errorHistory[epoch]);
                        if (epoch % (visualizationInterval * 2) == 0 || epoch == finalEpoch - 1) {
                            draw_current_decision_lines(Weights, bias, mean_params, std_params,
                                epoch, errorHistory[epoch], (epoch == finalEpoch - 1));
                        }
                        Application::DoEvents();
                    }
                }
            }           
        }

        textBox1->Text += "✓ EĞİTİM TAMAMLANDI (EarlyStopping: " + improvementThreshold + ")\r\n";
        textBox1->Text += "✓ EĞİTİM TAMAMLANDI (Epoch: " + finalEpoch + ")\r\n";
        textBox1->Text += "Son Hata: " + errorHistory[finalEpoch - 1].ToString("F6") + "\r\n";
        delete[] errorHistory;
        MessageBox::Show("Eğitim Tamamlandı!");
}
           
    private: System::Void testingToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
        if (numSample == 0 || mean_params == nullptr) {
            MessageBox::Show("Önce eğitim yapılmalı (Data veya Parametreler eksik)!");
            return;
        }

        if (isRegression) {
            Bitmap^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
            Graphics^ g = Graphics::FromImage(surface);
            g->Clear(Color::White);
            g->SmoothingMode = System::Drawing::Drawing2D::SmoothingMode::AntiAlias;

            Pen^ axisPen = gcnew Pen(Color::LightGray, 2);
            int cx = pictureBox1->Width / 2;
            int cy = pictureBox1->Height / 2;
            g->DrawLine(axisPen, cx, 0, cx, pictureBox1->Height);
            g->DrawLine(axisPen, 0, cy, pictureBox1->Width, cy);

            Pen^ regressionPen = gcnew Pen(Color::Red, 3);

            float* input1 = new float[inputDim];
            float* input2 = new float[inputDim];

            RegressionConfig config;
            config.inputDim = inputDim;
            config.outputDim = 1;

            for (int column = 0; column < pictureBox1->Width - 1; column += 2) {
                float x1 = float(column - cx);
                float x2 = float(column + 2 - cx);

                for (int d = 0; d < inputDim; d++) { input1[d] = 0.0f; input2[d] = 0.0f; }
                input1[0] = x1;
                input2[0] = x2;

                float y1, y2;

                if (useDynamicLayers && dynamicRegressionWeights != nullptr) {
                    y1 = EvalMLRegression(input1, dynamicRegressionWeights, mean_params, std_params, mean_y, std_y);
                    y2 = EvalMLRegression(input2, dynamicRegressionWeights, mean_params, std_params, mean_y, std_y);
                }
                else if (regressionWeights != nullptr) {
                    y1 = EvalSLRegression(input1, regressionWeights, config, mean_params, std_params, mean_y, std_y);
                    y2 = EvalSLRegression(input2, regressionWeights, config, mean_params, std_params, mean_y, std_y);
                }
                else { break; } 

                if (System::Single::IsNaN(y1) || System::Single::IsInfinity(y1) ||
                    System::Single::IsNaN(y2) || System::Single::IsInfinity(y2)) continue;

                int px1 = column;
                int py1 = cy - (int)y1;
                int px2 = column + 2;
                int py2 = cy - (int)y2;

                if (py1 > -5000 && py1 < 5000 && py2 > -5000 && py2 < 5000)
                    g->DrawLine(regressionPen, px1, py1, px2, py2);
            }

            delete[] input1;
            delete[] input2;

            for (int i = 0; i < numSample; i++) {
                int sx = (int)Samples[i * inputDim] + cx;
                int sy = cy - (int)targets[i];
                g->FillEllipse(gcnew SolidBrush(Color::Blue), sx - 4, sy - 4, 8, 8);
                g->DrawEllipse(gcnew Pen(Color::Black, 1), sx - 4, sy - 4, 8, 8);
            }

            pictureBox1->Image = surface;
            pictureBox1->Refresh();
            MessageBox::Show("Regresyon Testi Tamamlandı!");
        }
        else {
            Bitmap^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);

            float* x = new float[inputDim]; 
            int numClass;
            Color c;

            DynamicMultiLayerConfig dynConfig;
            NetworkConfig singleConfig;
            NetworkWeights singleWeights;

            if (useDynamicLayers) {
                dynConfig.inputDim = inputDim;
                dynConfig.outputDim = (class_count > 2) ? class_count : 1;
            }
            else {
                singleConfig.inputDim = inputDim;
                singleConfig.outputDim = (class_count > 2) ? class_count : 1;
                singleWeights.W_output = Weights; 
                singleWeights.b_output = bias;
            }

            for (int row = 0; row < pictureBox1->Height; row += 2) {
                for (int column = 0; column < pictureBox1->Width; column += 2) {


                    x[0] = (float)(column - (pictureBox1->Width / 2));
                    x[1] = (float)((pictureBox1->Height / 2) - row);

                    if (useDynamicLayers && dynamicWeights != nullptr) {
                        numClass = EvalMLClassify(x, dynamicWeights, dynConfig, mean_params, std_params);
                    }
                    else if (Weights != nullptr) {
                        numClass = EvalSLClassify(x, &singleWeights, singleConfig, mean_params, std_params);
                    }
                    else {
                        numClass = 0; 
                    }

                    switch (numClass) {
                    case 0: c = Color::FromArgb(150, 200, 200, 200); break; // Class 0 (Gri/Siyah)
                    case 1: c = Color::FromArgb(150, 255, 0, 0); break;     // Class 1 (Kırmızı)
                    case 2: c = Color::FromArgb(150, 0, 0, 255); break;     // Class 2 (Mavi)
                    case 3: c = Color::FromArgb(150, 0, 255, 0); break;     // Class 3 (Yeşil)
                    case 4: c = Color::FromArgb(150, 255, 255, 0); break;   // Class 4 (Sarı)
                    case 5: c = Color::FromArgb(150, 255, 0, 255); break;   // Class 5 (Mor)
                    default: c = Color::FromArgb(150, 0, 255, 255);
                    }

                    surface->SetPixel(column, row, c);
                    if (column + 1 < pictureBox1->Width) surface->SetPixel(column + 1, row, c);
                    if (row + 1 < pictureBox1->Height) surface->SetPixel(column, row + 1, c);
                    if (column + 1 < pictureBox1->Width && row + 1 < pictureBox1->Height)
                        surface->SetPixel(column + 1, row + 1, c);
                }
            }

            delete[] x; 

            Graphics^ g = Graphics::FromImage(surface);
            int cx = pictureBox1->Width / 2;
            int cy = pictureBox1->Height / 2;

            for (int i = 0; i < numSample; i++) {
                Pen^ pen;
                int label = (int)targets[i];
                switch (label) {
                case 0: pen = gcnew Pen(Color::Black, 3.0f); break;
                case 1: pen = gcnew Pen(Color::Red, 3.0f); break;
                case 2: pen = gcnew Pen(Color::Blue, 3.0f); break;
                case 3: pen = gcnew Pen(Color::Green, 3.0f); break;
                default: pen = gcnew Pen(Color::Yellow, 3.0f);
                }

                int temp_x = int(Samples[i * inputDim]) + cx;
                int temp_y = cy - int(Samples[i * inputDim + 1]);

                g->DrawLine(pen, temp_x - 6, temp_y, temp_x + 6, temp_y);
                g->DrawLine(pen, temp_x, temp_y - 6, temp_x, temp_y + 6);
            }

            pictureBox1->Image = surface;
            pictureBox1->Refresh();
            MessageBox::Show("Sınıflandırma Testi Tamamlandı!");
        }
    }

    private: System::Void UseMomentumCheckBox_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
        useMomentum = UseMomentumCheckBox->Checked;
        MomentumTextBox->Enabled = useMomentum;
        labelMomentum->Enabled = useMomentum;
    }
    private: System::Void Form1_Load(System::Object^ sender, System::EventArgs^ e) {
    }
    };
}
