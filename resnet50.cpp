#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <chrono>

TF_Session * sess_;
TF_Graph  *  graph_;


void CPUCalcSoftmax(const float* data, size_t size, float* result);
void TopK(const float* d, int size);


static int load_file(const std::string & fname, std::vector<char>& buf)
{
  std::ifstream fs(fname, std::ios::binary | std::ios::in);

  if(!fs.good())
  {
    std::cerr<<fname<<" does not exist"<<std::endl;
    return -1;
  }

  fs.seekg(0, std::ios::end);
  int fsize=fs.tellg();

  fs.seekg(0, std::ios::beg);
  buf.resize(fsize);
  fs.read(buf.data(),fsize);

  fs.close();

  return 0;

}


static TF_Session * load_graph(const char * frozen_fname, TF_Graph ** p_graph)
{
  TF_Status* s = TF_NewStatus();

  TF_Graph* graph = TF_NewGraph();

  std::vector<char> model_buf;

  if(load_file(frozen_fname,model_buf)<0)
    return nullptr;

  TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

  TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
  TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);

  if(TF_GetCode(s) != TF_OK)
  {
    printf("load graph failed!\n Error: %s\n",TF_Message(s));

    return nullptr;
  }

  TF_SessionOptions* sess_opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, sess_opts, s);
  assert(TF_GetCode(s) == TF_OK);


  TF_DeleteStatus(s);


  *p_graph=graph;

  return session;
}

/* To make tensor release happy...*/
static void dummy_deallocator(void* data, size_t len, void* arg)
{
}

void clean()
{
  TF_Status* s = TF_NewStatus();

  if(sess_)
  {
    TF_CloseSession(sess_,s);
    TF_DeleteSession(sess_,s);
  }

  if(graph_)
    TF_DeleteGraph(graph_);

  TF_DeleteStatus(s);
}

int load_model()
{

  std::string model_fname = "./resnet50.pb";  
  sess_= load_graph(model_fname.c_str(), &graph_);
  if(sess_== nullptr)
  {
    std::cout << "error";
    return -1;
  }
  return 0;
}

void CPUCalcSoftmax(const float* data, size_t size, float* result) {
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp(data[i]);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

void TopK(const float* d, int size) {
  std::priority_queue<std::pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(std::pair<float, int>(d[i], i));
  }

  std::pair<float, int> ki = q.top();

  std::cout << "idx: " << ki.second << " " << d[ki.second]<< "\n";
}

void inference(const char *image_path)
{
  int scale_h = 224;
  int scale_w = 224;
  cv::Mat image = cv::imread(image_path);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(scale_w, scale_h), 0, 0, cv::INTER_NEAREST);
  cv::Mat img2;
  resized.convertTo(img2, CV_32FC3);
  img2 = img2 - 128.0;

  /* tensorflow related*/
  TF_Status * s= TF_NewStatus();

  std::vector<TF_Output> input_names;
  std::vector<TF_Tensor*> input_values;

  TF_Operation* input_name=TF_GraphOperationByName(graph_, "input_1");

  input_names.push_back({input_name, 0});

  const int64_t dim[4] = {1,scale_h,scale_w,3};

  TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,dim,4,img2.ptr(),sizeof(float)*scale_w*scale_h*3,dummy_deallocator,nullptr);

  input_values.push_back(input_tensor);

  std::vector<TF_Output> output_names;

  TF_Operation* output_name = TF_GraphOperationByName(graph_,"fc1000/Softmax");
  output_names.push_back({output_name, 0});

  std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);

  for (int i=0; i<5; i++)
  {
    auto start = std::chrono::steady_clock::now();

    TF_SessionRun(sess_,nullptr,input_names.data(),input_values.data(),input_names.size(),
        output_names.data(),output_values.data(),output_names.size(),
        nullptr,0,nullptr,s);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in nanoseconds : "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
      << " ms\n";
  }

  assert(TF_GetCode(s) == TF_OK);

  /*retrieval the forward results*/
  const float * conf_data=(const float *)TF_TensorData(output_values[0]);
  int countnum = TF_TensorElementCount(output_values[0]);
  TopK(conf_data, countnum);

  TF_DeleteStatus(s);
  TF_DeleteTensor(output_values[0]);
  TF_DeleteTensor(input_tensor);
}


int main()
{
  load_model();
  inference("./img01.jpg");
  return 0;
}
