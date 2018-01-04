//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"Activation_DATA.hpp"
#include"Activation_FUNC.hpp"
#include"Activation_Base.h"

#include"Activation_GPU.cuh"
#include"Activation_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define BLOCK_SIZE	(32)

namespace
{
	//===========================
	// Leaky-ReLU
	//===========================
	__global__ void cuda_func_activation_LeakyReLU(const F32* i_lpInputBuffer, F32* o_lpOutputBuffer, F32 i_alpha, U32 i_bufferSize)
	{
		const U32 inputNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(inputNum >= i_bufferSize)	// ���򂷂邪������warp�����Ȃ̂ŁA�������x�ɉe���͂Ȃ��͂�...
			return;

		o_lpOutputBuffer[inputNum] = i_lpInputBuffer[inputNum] * ((i_lpInputBuffer[inputNum]>0) + i_alpha * (i_lpInputBuffer[inputNum]<=0));
	}
	__global__ void cuda_func_dactivation_LeakyReLU(const F32* i_lpOutputBuffer, const F32* i_lpDOutputBuffer, F32* o_lpOutputBuffer, F32 i_alpha, U32 i_bufferSize)
	{
		const U32 inputNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(inputNum >= i_bufferSize)	// ���򂷂邪������warp�����Ȃ̂ŁA�������x�ɉe���͂Ȃ��͂�...
			return;

		o_lpOutputBuffer[inputNum] = ((i_lpOutputBuffer[inputNum]>0) + i_alpha * (i_lpOutputBuffer[inputNum]<=0)) * i_lpDOutputBuffer[inputNum];
	}
}


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Activation_GPU::Activation_GPU(Gravisbell::GUID guid, Activation_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Activation_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	cudnnHandle		(NULL)
		,	activDesc		(NULL)
		,	inputTensorDesc	(NULL)
		,	outputTensorDesc	(NULL)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateActivationDescriptor(&activDesc);
	}
	/** �f�X�g���N�^ */
	Activation_GPU::~Activation_GPU()
	{
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(activDesc)			cudnnDestroyActivationDescriptor(activDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Activation_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Activation_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ILayerData& Activation_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Activation_GPU::GetLayerData()const
	{
		return this->layerData;
	}


	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �o�̓o�b�t�@���쐬
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			{
				int n = this->GetBatchSize();
				int c = this->GetOutputDataStruct().ch;
				int h = this->GetOutputDataStruct().z * this->GetOutputDataStruct().y;
				int w = this->GetOutputDataStruct().x;

				const int nDims = 4;
				int dimA[nDims] = {n, c, h, w};
				int strideA[nDims] = {c*h*w, h*w, w, 1};

				cudnnStatus_t err = cudnnSetTensorNdDescriptor(this->outputTensorDesc,
					CUDNN_DATA_FLOAT,
					4,
					dimA,
					strideA );

				if(err != 0)
					break;

				err = cudnnSetTensorNdDescriptor(this->inputTensorDesc,
					CUDNN_DATA_FLOAT,
					4,
					dimA,
					strideA );

				if(err != 0)
					break;
			}
			break;
		}


		// �������֐���ݒ�
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
		default:
			cudnnSetActivationDescriptor(activDesc,
										CUDNN_ACTIVATION_SIGMOID,
										CUDNN_PROPAGATE_NAN,
										0.0);
			break;

			// ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_ReLU:
			cudnnSetActivationDescriptor(activDesc,
										CUDNN_ACTIVATION_RELU,
										CUDNN_PROPAGATE_NAN,
										0.0);
			break;

			// Leaky-ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_LeakyReLU:
			break;

			// tanh
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_tanh:
			cudnnSetActivationDescriptor(activDesc,
										CUDNN_ACTIVATION_TANH,
										CUDNN_PROPAGATE_NAN,
										0.0);
			break;

			// SoftMax�n
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Activation_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			cudaMemcpy(o_lppOutputBuffer, i_lpInputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
			break;

		default:
			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
			// ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_ReLU:
			// tanh
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_tanh:
			{
				F32 alpha = 1.0f;
				F32 beta = 0.0f;
				cudnnActivationForward(
					this->cudnnHandle,
					this->activDesc,
					&alpha,
					inputTensorDesc,
					i_lpInputBuffer,
					&beta,
					outputTensorDesc,
					&o_lppOutputBuffer[0]);
			}
			break;

			// Leaky-ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_LeakyReLU:
			{
				U32 MAX_BUFFER_SIZE = 32768;
				U32 bufferSize = this->inputBufferCount * this->GetBatchSize();
				U32 remainingSize = bufferSize;
				while(remainingSize > 0)
				{
					U32 bufferCount = min(remainingSize, MAX_BUFFER_SIZE);
					dim3 grid((bufferCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
					dim3 block(BLOCK_SIZE, 1, 1);

					U32 offset = bufferSize - remainingSize;
					
					cuda_func_activation_LeakyReLU<<<grid, block>>>(
						&i_lpInputBuffer[offset],
						&o_lppOutputBuffer[offset],
						this->layerData.layerStructure.LeakyReLU_alpha,
						bufferCount);

					remainingSize = max(0, (S32)remainingSize-(S32)MAX_BUFFER_SIZE);
				}
			}	
			break;

			// softmax
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
			{
				F32 alpha = 1.0f;
				F32 beta = 0.0f;
				cudnnStatus_t err =	cudnnSoftmaxForward(
					this->cudnnHandle,
					CUDNN_SOFTMAX_ACCURATE,
					CUDNN_SOFTMAX_MODE_INSTANCE,
					&alpha,
					this->inputTensorDesc,
					i_lpInputBuffer,
					&beta,
					this->outputTensorDesc,
					&o_lppOutputBuffer[0]);
			}
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			{
				F32 alpha = 1.0f;
				F32 beta = 0.0f;
				cudnnSoftmaxForward(
					this->cudnnHandle,
					CUDNN_SOFTMAX_ACCURATE,
					CUDNN_SOFTMAX_MODE_CHANNEL,
					&alpha,
					this->inputTensorDesc,
					i_lpInputBuffer,
					&beta,
					this->outputTensorDesc,
					&o_lppOutputBuffer[0]);
			}
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Activation_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			switch(this->layerData.layerStructure.ActivationType)
			{
				// lenear
			case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
				cudaMemcpy(o_lppDInputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
				break;

			default:
				// Sigmoid
			case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid:
				// ReLU
			case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_ReLU:
				// tanh
			case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_tanh:
				{
					F32 alpha = 1.0f;
					F32 beta = 0.0f;
					cudnnActivationBackward(
						this->cudnnHandle,
						this->activDesc,
						&alpha,
						this->outputTensorDesc,	// �o�̓f�[�^�\��
						i_lppOutputBuffer,	// �o�̓f�[�^
						this->outputTensorDesc,
						i_lppDOutputBuffer,	// �o�͌덷
						this->inputTensorDesc,
						i_lppInputBuffer,	// ����
						&beta,
						this->inputTensorDesc,
						o_lppDInputBuffer	// ���͌덷
						);
				}
				break;

					// Leaky-ReLU
				case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_LeakyReLU:
					{
						U32 MAX_BUFFER_SIZE = 32768;
						U32 bufferSize = this->inputBufferCount * this->GetBatchSize();
						U32 remainingSize = bufferSize;
						while(remainingSize > 0)
						{
							U32 bufferCount = min(remainingSize, MAX_BUFFER_SIZE);
							dim3 grid((bufferCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
							dim3 block(BLOCK_SIZE, 1, 1);

							U32 offset = bufferSize - remainingSize;

							cuda_func_dactivation_LeakyReLU<<<grid, block>>>(
								&i_lppOutputBuffer[offset],
								&i_lppDOutputBuffer[offset],
								&o_lppDInputBuffer[offset],
								this->layerData.layerStructure.LeakyReLU_alpha,
								bufferCount);

							remainingSize = max(0, (S32)remainingSize-(S32)MAX_BUFFER_SIZE);
						}
					}
					break;

				// softmax
				case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
					{
						F32 alpha = 1.0f;
						F32 beta = 0.0f;
						cudnnSoftmaxBackward(
							this->cudnnHandle,
							CUDNN_SOFTMAX_ACCURATE,
							CUDNN_SOFTMAX_MODE_INSTANCE,
							&alpha,
							this->outputTensorDesc,
							i_lppOutputBuffer,
							this->outputTensorDesc,
							i_lppDOutputBuffer,
							&beta,
							this->inputTensorDesc,
							o_lppDInputBuffer
							);
					}
					break;
				case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
					{
						F32 alpha = 1.0f;
						F32 beta = 0.0f;
						cudnnSoftmaxBackward(
							this->cudnnHandle,
							CUDNN_SOFTMAX_ACCURATE,
							CUDNN_SOFTMAX_MODE_CHANNEL,
							&alpha,
							this->outputTensorDesc,
							i_lppOutputBuffer,
							this->outputTensorDesc,
							i_lppDOutputBuffer,
							&beta,
							this->inputTensorDesc,
							o_lppDInputBuffer
							);
					}
					break;

				case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
				case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
				case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
					cudaMemcpy(o_lppDInputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
					break;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Activation_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
