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

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)
#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector,chSize)	POSITION_TO_OFFSET(inX, inY, inZ, inCh, vector.x,    vector.y,    vector.z,    chSize)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Convolution_CPU::Convolution_CPU(Gravisbell::GUID guid, Convolution_LayerData_CPU& i_layerData)
		:	Convolution_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount						(0)		/**< �j���[������ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	m_lppInputBuffer				(NULL)	/**< ���Z���̓��̓f�[�^ */
		,	onUseDropOut					(false)
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
	Convolution_LayerData_Base& Convolution_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Convolution_LayerData_Base& Convolution_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Convolution_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		return this->layerData.WriteToBuffer(o_lpBuffer);
	}


	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ���͍����o�b�t�@���쐬
		this->lpDInputBuffer.resize(this->batchSize);
		this->lppBatchDInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpDInputBuffer[batchNum].resize(this->inputBufferCount);
			this->lppBatchDInputBuffer[batchNum] = &this->lpDInputBuffer[batchNum][0];
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_CPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

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
		if(this->layerData.lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppNeuron[0].size() != this->filterSize * this->layerData.inputDataStruct.ch)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;


		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum][0];
		}
		
		// �p�f�B���O��̓��̓o�b�t�@���쐬
		this->paddingInputDataStruct.x  = this->layerData.convolutionCount.x + this->layerData.layerStructure.FilterSize.x - 1;
		this->paddingInputDataStruct.y  = this->layerData.convolutionCount.y + this->layerData.layerStructure.FilterSize.y - 1;
		this->paddingInputDataStruct.z  = this->layerData.convolutionCount.z + this->layerData.layerStructure.FilterSize.z - 1;
		this->paddingInputDataStruct.ch = this->layerData.inputDataStruct.ch;
		this->lpPaddingInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpPaddingInputBuffer[batchNum].resize(this->paddingInputDataStruct.GetDataCount(), 0.0f);
		}

		// �h���b�v�A�E�g�����𖢎g�p�ɕύX
		this->onUseDropOut = false;
		this->lppDropOutBuffer.clear();


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// �j���[����/�o�C�A�X�̌덷���ꎞ�ۑ�����o�b�t�@���쐬
		if(lpDBias.empty() || lppDNeuron.empty())
		{
			this->lpDBias.resize(this->neuronCount);
			this->lppDNeuron.resize(this->neuronCount);
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				this->lppDNeuron[neuronNum].resize(this->filterSize * this->layerData.inputDataStruct.ch);
		}
		// �h���b�v�A�E�g
		{
			S32 dropOutRate = (S32)(this->layerData.layerStructure.DropOut * RAND_MAX);

			if(dropOutRate > 0)
			{
				this->onUseDropOut = true;
				if(this->lppDropOutBuffer.empty())
				{
					// �o�b�t�@�̊m��
					this->lppDropOutBuffer.resize(this->neuronCount);
					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
						this->lppDropOutBuffer[neuronNum].resize(this->filterSize * this->layerData.inputDataStruct.ch);
				}

				// �o�b�t�@��1or0�����
				// 1 : DropOut���Ȃ�
				// 0 : DropOut����
				for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					for(U32 inputNum=0; inputNum<this->filterSize * this->layerData.inputDataStruct.ch; inputNum++)
					{
						if(rand() < dropOutRate)	// �h���b�v�A�E�g����
							this->lppDropOutBuffer[neuronNum][inputNum] = 0.0f;
						else
							this->lppDropOutBuffer[neuronNum][inputNum] = 1.0f;
					}
				}

				// �h���b�v�A�E�g�������������j���[�������쐬����
				if(this->lppDropOutNeuron.empty())
				{
					// �o�b�t�@�̊m��
					this->lppDropOutNeuron.resize(this->neuronCount);
					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
						this->lppDropOutNeuron[neuronNum].resize(this->filterSize * this->layerData.inputDataStruct.ch);
				}
			}
			else
			{
				this->onUseDropOut = false;
				this->lppDropOutBuffer.clear();
				this->lppDropOutNeuron.clear();
			}
		}
		// �w�K�W��
		{
			auto pItem = dynamic_cast<const Gravisbell::SettingData::Standard::IItem_Float*>(data.GetItemByID(L"LearnCoeff"));
			if(pItem)
				this->learnData.LearnCoeff = pItem->GetValue();
			else
				this->learnData.LearnCoeff = 1.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Convolution_CPU::PreProcessCalculateLoop()
	{
		this->onUseDropOut = false;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Convolution_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		// �h���b�v�A�E�g�����s����ꍇ�A�j���[�������h���b�v�A�E�g�����h���b�v�A�E�g�p�o�b�t�@�Ɉړ�
		if(this->onUseDropOut)
		{
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				for(U32 i=0; i<this->filterSize*this->layerData.inputDataStruct.ch; i++)
				{
					this->lppDropOutNeuron[neuronNum][i] = this->lppDropOutBuffer[neuronNum][i] * this->layerData.lppNeuron[neuronNum][i];
				}
			}
		}
		const std::vector<std::vector<F32>>& lppNeuron = this->onUseDropOut ? this->lppDropOutNeuron : this->layerData.lppNeuron;


		// �p�f�B���O��̓��̓o�b�t�@�Ƀf�[�^���ڂ�
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
			{
				for(U32 paddingZ=0; paddingZ<this->paddingInputDataStruct.z; paddingZ++)
				{
					S32 inputZ = paddingZ - this->layerData.layerStructure.Padding.z;
					if((U32)inputZ>=this->layerData.inputDataStruct.z)
						continue;

					for(U32 paddingY=0; paddingY<this->paddingInputDataStruct.y; paddingY++)
					{
						S32 inputY = paddingY - this->layerData.layerStructure.Padding.y;
						if((U32)inputY>=this->layerData.inputDataStruct.y)
							continue;

						for(U32 paddingX=0; paddingX<this->paddingInputDataStruct.x; paddingX++)
						{
							S32 inputX = paddingX - this->layerData.layerStructure.Padding.x;
							if((U32)inputX>=this->layerData.inputDataStruct.x)
								continue;

							S32 paddingOffset = POSITION_TO_OFFSET_STRUCT(paddingX, paddingY, paddingZ, chNum, this->paddingInputDataStruct);
							S32 inputOffset   = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, chNum, this->layerData.inputDataStruct);

							this->lpPaddingInputBuffer[batchNum][paddingOffset] = i_lppInputBuffer[batchNum][inputOffset];
						}
					}
				}
			}
		}

		// ��݂��݌�������
		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 neuronNum=0; neuronNum<(U32)this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				for(S32 convZ=0; convZ<this->layerData.convolutionCount.z; convZ++)
				{
					for(S32 convY=0; convY<this->layerData.convolutionCount.y; convY++)
					{
						for(S32 convX=0; convX<this->layerData.convolutionCount.x; convX++)
						{
							U32 outputOffset = POSITION_TO_OFFSET_VECTOR(convX,convY,convZ,neuronNum, this->layerData.convolutionCount, this->layerData.layerStructure.Output_Channel);

							// �ꎞ�ۑ��p�̃o�b�t�@���쐬
							F32 tmp = 0.0f;

							// �t�B���^����������
							for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)(convZ * this->layerData.layerStructure.Stride.z + filterZ);

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)(convY * this->layerData.layerStructure.Stride.y + filterY);

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)(convX * this->layerData.layerStructure.Stride.x + filterX);

											const S32 inputOffset  = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->paddingInputDataStruct);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize, this->layerData.inputDataStruct.ch);

											tmp += lppNeuron[neuronNum][filterOffset] * this->lpPaddingInputBuffer[batchNum][inputOffset];
										}
									}
								}
							}
							// �o�C�A�X��ǉ�
							tmp += this->layerData.lpBias[neuronNum];

							// ���Z���̃h���b�v�A�E�g���������s
							if(!this->onUseDropOut && this->layerData.layerStructure.DropOut > 0.0f)
								tmp *= (1.0f - this->layerData.layerStructure.DropOut);


							// �v�Z���ʂ��i�[����
							this->lpOutputBuffer[batchNum][outputOffset] = tmp;

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
		return &this->lppBatchOutputBuffer[0];
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

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpOutputBuffer[batchNum], this->lppBatchOutputBuffer[batchNum], sizeof(F32)*outputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Convolution_CPU::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		this->m_lppDOutputBuffer = i_lppDOutputBufferPrev;

		// �g�p����j���[�����f�[�^��I������
		const std::vector<std::vector<F32>>& lppNeuron = this->onUseDropOut ? this->lppDropOutNeuron : this->layerData.lppNeuron;


		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			// ���͌덷�o�b�t�@��������
			memset(&this->lpDInputBuffer[batchNum][0], 0, this->lpDInputBuffer[batchNum].size()*sizeof(F32));

			// ���͌덷�v�Z
			for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
			{
				// �j���[�����ƃo�C�A�X�̕ω��ʂ�������
				this->lpDBias[neuronNum] = 0.0f;
				memset(&this->lppDNeuron[neuronNum][0], 0, this->filterSize * this->layerData.inputDataStruct.ch * sizeof(F32));


				for(S32 convZ=0; convZ<this->layerData.convolutionCount.z; convZ++)
				{
					for(S32 convY=0; convY<this->layerData.convolutionCount.y; convY++)
					{
						for(S32 convX=0; convX<this->layerData.convolutionCount.x; convX++)
						{
							U32 outputOffet = POSITION_TO_OFFSET_VECTOR(convX,convY,convZ,neuronNum, this->layerData.convolutionCount, this->layerData.layerStructure.Output_Channel);
							F32 dOutput = this->m_lppDOutputBuffer[batchNum][outputOffet];

							// �t�B���^����������
							for(U32 chNum=0; chNum<this->layerData.inputDataStruct.ch; chNum++)
							{
								for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
								{
									const S32 inputZ = (S32)((convZ - this->layerData.convolutionStart.z) * this->layerData.layerStructure.Stride.z + filterZ);
									if((U32)inputZ>=this->layerData.inputDataStruct.z)
										continue;

									for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
									{
										const S32 inputY = (S32)((convY - this->layerData.convolutionStart.y) * this->layerData.layerStructure.Stride.y + filterY);
										if((U32)inputY>=this->layerData.inputDataStruct.y)
											continue;

										for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
										{
											const S32 inputX = (S32)((convX - this->layerData.convolutionStart.x) * this->layerData.layerStructure.Stride.x + filterX);
											if((U32)inputX>=this->layerData.inputDataStruct.x)
												continue;

											const S32 inputOffset  = POSITION_TO_OFFSET_STRUCT(inputX,inputY,inputZ, chNum, this->layerData.inputDataStruct);
											const S32 filterOffset = POSITION_TO_OFFSET_VECTOR(filterX, filterY, filterZ, chNum, this->layerData.layerStructure.FilterSize, this->layerData.inputDataStruct.ch);


											this->lpDInputBuffer[batchNum][inputOffset] += lppNeuron[neuronNum][filterOffset] * dOutput;

											// �j���[�����̏d�ݕω��ʂ�ǉ�
											this->lppDNeuron[neuronNum][filterOffset] += this->m_lppInputBuffer[batchNum][inputOffset] * dOutput;
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


	/** �w�K���������C���[�ɔ��f������.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		�o�͌덷�����A���͌덷�����͒��O��CalculateLearnError�̒l���Q�Ƃ���. */
	ErrorCode Convolution_CPU::ReflectionLearnError(void)
	{
		U32 filterDataSize = this->layerData.layerStructure.FilterSize.x * this->layerData.layerStructure.FilterSize.y * this->layerData.layerStructure.FilterSize.z * this->layerData.inputDataStruct.ch;

		for(S32 neuronNum=0; neuronNum<this->layerData.layerStructure.Output_Channel; neuronNum++)
		{
			// �o�C�A�X�X�V
			this->layerData.lpBias[neuronNum] += this->lpDBias[neuronNum] * this->learnData.LearnCoeff;

			// �e�j���[�������X�V
			for(U32 filterOffset=0; filterOffset<filterDataSize; filterOffset++)
			{
				this->layerData.lppNeuron[neuronNum][filterOffset] += this->lppDNeuron[neuronNum][filterOffset] * this->learnData.LearnCoeff;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Convolution_CPU::GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode Convolution_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpDInputBuffer[batchNum], this->lppBatchDInputBuffer[batchNum], sizeof(F32)*inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
