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

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)
#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector,chSize)	POSITION_TO_OFFSET(inX, inY, inZ, inCh, vector.x,    vector.y,    vector.z,    chSize)


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

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDOutputBuffer.resize(this->GetBatchSize());
		
		// ���͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDInputBuffer.resize(this->GetBatchSize());

		// �j���[����/�o�C�A�X�̌덷���ꎞ�ۑ�����o�b�t�@���쐬
		this->lpDBias.resize(this->layerData.lpBias.size());
		this->lpDNeuron.resize(this->layerData.lpNeuron.size());

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

		// ���̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->GetBatchSize() * this->outputBufferCount);
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum * this->outputBufferCount];
		}


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
	ErrorCode Convolution_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			U32 PROCTIME_MAX = 5;			// ���s�ő�l
			F32	VARIANCE_TOLERANCE = 0.1f;	// ���U����(���e�͈�)

			U32 procTime = 0;
			do
			{
				// ���Z�����s
				ErrorCode err = this->Calculate();
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
			ErrorCode err = this->Calculate();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	ErrorCode Convolution_CPU::Calculate()
	{
		// ��݂��݌�������
		for(unsigned int batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 neuronNum=0; neuronNum<(U32)this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(S32 convZ=0; convZ<this->GetOutputDataStruct().z; convZ++)
				{
					for(S32 convY=0; convY<this->GetOutputDataStruct().y; convY++)
					{
						for(S32 convX=0; convX<this->GetOutputDataStruct().x; convX++)
						{
							U32 outputOffset = POSITION_TO_OFFSET_STRUCT(convX,convY,convZ,neuronNum, this->GetOutputDataStruct());

							// �ꎞ�ۑ��p�̃o�b�t�@���쐬
							F32 tmp = 0.0f;

							// �t�B���^����������
							for(U32 chNum=0; chNum<this->GetInputDataStruct().ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ - this->layerData.layerStructure.Padding.z);
									if((U32)inputZ >= this->GetInputDataStruct().z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY - this->layerData.layerStructure.Padding.y);
										if((U32)inputY >= this->GetInputDataStruct().y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX - this->layerData.layerStructure.Padding.x);
											if((U32)inputX >= this->GetInputDataStruct().x)
												continue;

											const S32 inputOffset  = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->GetInputDataStruct());
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize, this->layerData.inputDataStruct.ch);

											tmp += this->layerData.lppNeuron[neuronNum][filterOffset] * this->m_lppInputBuffer[batchNum][inputOffset];
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


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Convolution_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode Convolution_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

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
	ErrorCode Convolution_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		
		// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpDInputBuffer = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
			
			// ���͌덷�o�b�t�@��������
			memset(this->m_lpDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());
		}

		
		// �j���[�����ƃo�C�A�X�̕ω��ʂ�������
		memset(&this->lpDBias[0], 0, sizeof(F32)*this->lpDBias.size());
		memset(&this->lpDNeuron[0], 0, sizeof(F32)*this->lpDNeuron.size());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			// ���͌덷�v�Z
			for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(S32 convZ=0; convZ<this->GetOutputDataStruct().z; convZ++)
				{
					for(S32 convY=0; convY<this->GetOutputDataStruct().y; convY++)
					{
						for(S32 convX=0; convX<this->GetOutputDataStruct().x; convX++)
						{
							U32 outputOffet = POSITION_TO_OFFSET_STRUCT(convX,convY,convZ,neuronNum, this->GetOutputDataStruct());
							F32 dOutput = this->m_lppDOutputBuffer[batchNum][outputOffet];

							// �t�B���^����������
							for(U32 chNum=0; chNum<this->GetInputDataStruct().ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)((convZ * this->layerData.layerStructure.Stride.z) - this->layerData.layerStructure.Padding.z + filterZ);
									if((U32)inputZ>=this->GetInputDataStruct().z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)((convY * this->layerData.layerStructure.Stride.y) - this->layerData.layerStructure.Padding.y + filterY);
										if((U32)inputY>=this->GetInputDataStruct().y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)((convX * this->layerData.layerStructure.Stride.x) - this->layerData.layerStructure.Padding.x + filterX);
											if((U32)inputX>=this->GetInputDataStruct().x)
												continue;

											const S32 inputOffset  = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->GetInputDataStruct());
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize, this->layerData.inputDataStruct.ch);

											if(this->m_lpDInputBuffer)
												this->m_lppDInputBuffer[batchNum][inputOffset] += this->layerData.lppNeuron[neuronNum][filterOffset] * dOutput;

											// �j���[�����̏d�ݕω��ʂ�ǉ�
											this->lpDNeuron[neuronNum*this->filterSize*this->GetInputDataStruct().ch + filterOffset] += this->m_lppInputBuffer[batchNum][inputOffset] * dOutput;
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
	ErrorCode Convolution_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// �w�K�����̔��f
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(&this->layerData.lpBias[0], &this->lpDBias[0]);
		if(this->layerData.m_pOptimizer_neuron)
			this->layerData.m_pOptimizer_neuron->UpdateParameter(&this->layerData.lpNeuron[0], &this->lpDNeuron[0]);


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Convolution_CPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode Convolution_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
