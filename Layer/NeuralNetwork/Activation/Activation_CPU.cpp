//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"Activation_DATA.hpp"
#include"Activation_FUNC.hpp"
#include"Activation_Base.h"

#include"Activation_CPU.h"
#include"Activation_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Activation_CPU::Activation_CPU(Gravisbell::GUID guid, Activation_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Activation_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	func_activation					(&Activation_CPU::func_activation_sigmoid)
		,	func_dactivation				(&Activation_CPU::func_dactivation_sigmoid)
	{
	}
	/** �f�X�g���N�^ */
	Activation_CPU::~Activation_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Activation_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Activation_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ILayerData& Activation_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Activation_CPU::GetLayerData()const
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
	ErrorCode Activation_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ���͌덷/�o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppDOutputBuffer.resize(this->GetBatchSize(), NULL);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_CPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ����/�o�̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppOutputBuffer.resize(this->GetBatchSize(), NULL);


		// �������֐���ݒ�
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			this->func_activation  = &Activation_CPU::func_activation_lenear;
			this->func_dactivation = &Activation_CPU::func_dactivation_lenear;
			break;

			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid:
		default:
			this->func_activation  = &Activation_CPU::func_activation_sigmoid;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
			this->func_activation  = &Activation_CPU::func_activation_sigmoid_crossEntropy;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid_crossEntropy;
			break;

			// ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_ReLU:
			this->func_activation  = &Activation_CPU::func_activation_ReLU;
			this->func_dactivation = &Activation_CPU::func_dactivation_ReLU;
			break;

			// Leakey-ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_LeakyReLU:
			this->func_activation  = &Activation_CPU::func_activation_LeakyReLU;
			this->func_dactivation = &Activation_CPU::func_dactivation_LeakyReLU;
			break;

			// SoftMax�n
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
			this->func_activation  = &Activation_CPU::func_activation_SoftMax;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->func_activation  = &Activation_CPU::func_activation_SoftMax;
			this->func_dactivation = &Activation_CPU::func_dactivation_sigmoid_crossEntropy;
			break;
		}

		// ���Z�p�̃o�b�t�@���m��
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->lpCalculateSum.resize(this->GetInputDataStruct().z * this->GetInputDataStruct().y * this->GetInputDataStruct().x);
			break;
		default:
			this->lpCalculateSum.clear();
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Activation_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			memcpy(o_lppOutputBuffer, i_lppInputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());
			break;

		default:
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					// ������
					this->m_lppOutputBuffer[batchNum][inputNum] = (this->*func_activation)(this->m_lppInputBuffer[batchNum][inputNum]);
				}
			}
			break;
		}

		// softMax�����s����
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					// ���v�l���Z�o
					F32 sum = 0.0f;
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						sum += this->m_lppOutputBuffer[batchNum][outputNum];
					}

					// �l�����v�l�Ŋ���
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						if(sum == 0.0f)
							this->m_lppOutputBuffer[batchNum][outputNum] = 1.0f / this->outputBufferCount;
						else
							this->m_lppOutputBuffer[batchNum][outputNum] /= sum;
					}
				}
			}
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			{
				U32 chSize = this->GetInputDataStruct().z * this->GetInputDataStruct().y * this->GetInputDataStruct().x;

				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					// �ꎞ�o�b�t�@�N���A
					memset(&this->lpCalculateSum[0], 0, this->lpCalculateSum.size()*sizeof(F32));

					// ���v�l���Z�o
					for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
					{
						for(U32 z=0; z<this->GetInputDataStruct().z; z++)
						{
							for(U32 y=0; y<this->GetInputDataStruct().y; y++)
							{
								for(U32 x=0; x<this->GetInputDataStruct().x; x++)
								{
									U32 offset = (((((ch*this->GetInputDataStruct().z+z)*this->GetInputDataStruct().y)+y)*this->GetInputDataStruct().x)+x);

									this->lpCalculateSum[offset] += this->m_lppOutputBuffer[batchNum][ch*chSize + offset];
								}
							}
						}
					}

					// ���v�l�Ŋ���
					for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
					{
						for(U32 z=0; z<this->GetInputDataStruct().z; z++)
						{
							for(U32 y=0; y<this->GetInputDataStruct().y; y++)
							{
								for(U32 x=0; x<this->GetInputDataStruct().x; x++)
								{
									U32 offset = (((((ch*this->GetInputDataStruct().z+z)*this->GetInputDataStruct().y)+y)*this->GetInputDataStruct().x)+x);

									this->m_lppOutputBuffer[batchNum][ch*chSize + offset] /= this->lpCalculateSum[offset];
								}
							}
						}
					}
				}
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
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz���[GetOutputDataCount()]�z��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Activation_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
//				this->m_lppInputBuffer[batchNum]   = &i_lppInputBuffer[batchNum * this->inputBufferCount];
				this->m_lppDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->m_lppOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
				this->m_lppDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
			}

			// ���͌덷���v�Z
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					this->m_lppDInputBuffer[batchNum][inputNum] = (this->*func_dactivation)(this->m_lppOutputBuffer[batchNum][inputNum]) * this->m_lppDOutputBuffer[batchNum][inputNum];
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Activation_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}



	//================================
	// �������֐�
	//================================
	// lenear�n
	F32 Activation_CPU::func_activation_lenear(F32 x)
	{
		return x;
	}
	F32 Activation_CPU::func_dactivation_lenear(F32 x)
	{
		return 1;
	}

	// sigmoid�n
	F32 Activation_CPU::func_activation_sigmoid(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 Activation_CPU::func_dactivation_sigmoid(F32 x)
	{
		return x * (1.0f - x);
	}

	F32 Activation_CPU::func_activation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 Activation_CPU::func_dactivation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f;
	}

	// ReLU�n
	F32 Activation_CPU::func_activation_ReLU(F32 x)
	{
		return x * (x > 0.0f);
	}
	F32 Activation_CPU::func_dactivation_ReLU(F32 x)
	{
		return 1.0f * (x > 0.0f);
	}

	// Leakey-ReLU�n
	F32 Activation_CPU::func_activation_LeakyReLU(F32 x)
	{
		return x * ( (x>0.0f) + this->layerData.layerStructure.LeakyReLU_alpha * (x<=0.0f) );
	}
	F32 Activation_CPU::func_dactivation_LeakyReLU(F32 x)
	{
		return (x>0.0f) + this->layerData.layerStructure.LeakyReLU_alpha * (x<=0.0f);
	}

	// tanh�n
	F32 Activation_CPU::func_activation_tanh(F32 x)
	{
		return tanh(x);
	}
	F32 Activation_CPU::func_dactivation_tanh(F32 x)
	{
		return 1.0f - x*x;
	}

	// SoftMax�n
	F32 Activation_CPU::func_activation_SoftMax(F32 x)
	{
		return min(FLT_MAX, (F32)exp(x));	// ���ς͕ʂɍs��
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
