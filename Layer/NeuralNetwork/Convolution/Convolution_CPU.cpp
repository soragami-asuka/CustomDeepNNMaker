//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌������C���[
// CPU�����p
//======================================
#include"stdafx.h"

#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"

#include"Convolution_CPU.h"
#include"Convolution_LayerData_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector)	Gravisbell::CalculateOffset((vector).x,    (vector).y,    (vector).z, inCh, inX, inY, inZ)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Convolution_CPU::Convolution_CPU(Gravisbell::GUID guid, Convolution_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Convolution_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount						(0)		/**< �j���[������ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
	}
	/** �f�X�g���N�^ */
	Convolution_CPU::~Convolution_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Convolution_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Convolution_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ILayerData& Convolution_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Convolution_CPU::GetLayerData()const
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
	ErrorCode Convolution_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �j���[����/�o�C�A�X�̌덷���ꎞ�ۑ�����o�b�t�@���쐬
		this->lpDBias.resize(this->layerData.lpBias.size());
		this->lpDNeuron.resize(this->layerData.lpNeuron.size());

		// ���͌덷/�o�͌덷�o�b�t�@���쐬
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_CPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// �j���[���������m�F
		this->neuronCount = this->layerData.layerStructure.Output_Channel;
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �t�B���^�T�C�Y�m�F
		this->filterSize = this->layerData.layerStructure.FilterSize.x * this->layerData.layerStructure.FilterSize.y * this->layerData.layerStructure.FilterSize.z;
		if(this->filterSize == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �j���[�����o�b�t�@�̃T�C�Y�m�F
		if(this->layerData.lpNeuron.size() != this->neuronCount * this->filterSize * this->GetInputDataStruct().ch)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@���쐬
		this->lppBatchInputBuffer.resize(this->GetBatchSize());
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Convolution_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			U32 PROCTIME_MAX = 5;			// ���s�ő�l
			F32	VARIANCE_TOLERANCE = 0.1f;	// ���U����(���e�͈�)

			U32 procTime = 0;
			do
			{
				// ���Z�����s
				ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				// �o�͂̕��U�����߂�
				F32 variance = 0.0f;
				F32 average  = 0.0f;
				{
					// ���ς����߂�
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
						{
							average += this->lppBatchOutputBuffer[batchNum][outputNum];
						}
					}
					average /= (this->outputBufferCount * this->GetBatchSize());

					// ���U�����߂�
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
						{
							variance += (this->lppBatchOutputBuffer[batchNum][outputNum] - average) * (this->lppBatchOutputBuffer[batchNum][outputNum] - average);
						}
					}
					variance /= (this->outputBufferCount * this->GetBatchSize());
				}

				if( abs(variance - 1.0f) < VARIANCE_TOLERANCE)
					break;

				// �W���΍��ŏd�݂������čX�V����
				F32 deviation = sqrtf(variance);
				{
					for(U32 neuronNum=0; neuronNum<this->layerData.lpNeuron.size(); neuronNum++)
					{
						this->layerData.lpNeuron[neuronNum] /= deviation;
					}
					for(U32 neuronNum=0; neuronNum<this->layerData.lpBias.size(); neuronNum++)
					{
						this->layerData.lpBias[neuronNum] /= deviation;
					}
				}

				procTime++;
			}while(procTime < PROCTIME_MAX);
		}
		else
		{
			ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	ErrorCode Convolution_CPU::Calculate_base(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
			this->lppBatchInputBuffer[batchNum] = &i_lppInputBuffer[batchNum * this->inputBufferCount];
		}

		// ��݂��݌�������
		for(unsigned int batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 neuronNum=0; neuronNum<(U32)this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(U32 convZ=0; convZ<this->GetOutputDataStruct().z; convZ++)
				{
					for(U32 convY=0; convY<this->GetOutputDataStruct().y; convY++)
					{
						for(U32 convX=0; convX<this->GetOutputDataStruct().x; convX++)
						{
							U32 outputOffset = this->GetOutputDataStruct().POSITION_TO_OFFSET(convX,convY,convZ,neuronNum);

							// �ꎞ�ۑ��p�̃o�b�t�@���쐬
							F32 tmp = 0.0f;

							// �t�B���^����������
							for(U32 chNum=0; chNum<this->GetInputDataStruct().ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ*this->layerData.layerStructure.Dilation.z - this->layerData.layerStructure.Padding.z);
									if((U32)inputZ >= this->GetInputDataStruct().z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY*this->layerData.layerStructure.Dilation.y - this->layerData.layerStructure.Padding.y);
										if((U32)inputY >= this->GetInputDataStruct().y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX*this->layerData.layerStructure.Dilation.x - this->layerData.layerStructure.Padding.x);
											if((U32)inputX >= this->GetInputDataStruct().x)
												continue;

											const S32 inputOffset  = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,inputY,inputZ, chNum);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize);

											tmp += this->layerData.lppNeuron[neuronNum][filterOffset] * this->lppBatchInputBuffer[batchNum][inputOffset];
										}
									}
								}
							}
							// �o�C�A�X��ǉ�
							tmp += this->layerData.lpBias[neuronNum];

							// �v�Z���ʂ��i�[����
							this->lppBatchOutputBuffer[batchNum][outputOffset] = tmp;

						}
					}
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Convolution_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
			this->lppBatchInputBuffer[batchNum] = &i_lppInputBuffer[batchNum * this->inputBufferCount];
		}

		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		
		// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->lppBatchDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
			
			// ���͌덷�o�b�t�@��������
			memset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());
		}

		
		// �j���[�����ƃo�C�A�X�̕ω��ʂ�������
		memset(&this->lpDBias[0], 0, sizeof(F32)*this->lpDBias.size());
		memset(&this->lpDNeuron[0], 0, sizeof(F32)*this->lpDNeuron.size());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			// ���͌덷�v�Z
			for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(U32 convZ=0; convZ<this->GetOutputDataStruct().z; convZ++)
				{
					for(U32 convY=0; convY<this->GetOutputDataStruct().y; convY++)
					{
						for(U32 convX=0; convX<this->GetOutputDataStruct().x; convX++)
						{
							U32 outputOffet = this->GetOutputDataStruct().POSITION_TO_OFFSET(convX,convY,convZ,neuronNum);
							F32 dOutput = this->lppBatchDOutputBuffer[batchNum][outputOffet];

							// �t�B���^����������
							for(U32 chNum=0; chNum<this->GetInputDataStruct().ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)((convZ * this->layerData.layerStructure.Stride.z) - this->layerData.layerStructure.Padding.z + filterZ*this->layerData.layerStructure.Dilation.z);
									if((U32)inputZ>=this->GetInputDataStruct().z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)((convY * this->layerData.layerStructure.Stride.y) - this->layerData.layerStructure.Padding.y + filterY*this->layerData.layerStructure.Dilation.y);
										if((U32)inputY>=this->GetInputDataStruct().y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)((convX * this->layerData.layerStructure.Stride.x) - this->layerData.layerStructure.Padding.x + filterX*this->layerData.layerStructure.Dilation.x);
											if((U32)inputX>=this->GetInputDataStruct().x)
												continue;

											const S32 inputOffset  = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,inputY,inputZ, chNum);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize);

											if(o_lppDInputBuffer)
												this->lppBatchDInputBuffer[batchNum][inputOffset] += this->layerData.lppNeuron[neuronNum][filterOffset] * dOutput;

											// �j���[�����̏d�ݕω��ʂ�ǉ�
											this->lpDNeuron[neuronNum*this->filterSize*this->GetInputDataStruct().ch + filterOffset] += this->lppBatchInputBuffer[batchNum][inputOffset] * dOutput;
										}
									}
								}
							}

							// �o�C�A�X�̏d�ݕω��ʂ�ǉ�
							this->lpDBias[neuronNum] += dOutput;

						}
					}
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Convolution_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// �w�K�����̔��f
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(&this->layerData.lpBias[0], &this->lpDBias[0]);
		if(this->layerData.m_pOptimizer_neuron)
			this->layerData.m_pOptimizer_neuron->UpdateParameter(&this->layerData.lpNeuron[0], &this->lpDNeuron[0]);


		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
