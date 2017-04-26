//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"Pooling_DATA.hpp"
#include"Pooling_FUNC.hpp"
#include"Pooling_Base.h"

#include"Pooling_CPU.h"
#include"Pooling_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Pooling_CPU::Pooling_CPU(Gravisbell::GUID guid, Pooling_LayerData_CPU& i_layerData)
		:	Pooling_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	m_lppInputBuffer				(NULL)	/**< ���Z���̓��̓f�[�^ */
		,	m_lppDOutputBufferPrev			(NULL)	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */
	{
	}
	/** �f�X�g���N�^ */
	Pooling_CPU::~Pooling_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Pooling_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Pooling_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Pooling_LayerData_Base& Pooling_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Pooling_LayerData_Base& Pooling_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Pooling_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
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
	ErrorCode Pooling_CPU::PreProcessLearn(unsigned int batchSize)
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
	ErrorCode Pooling_CPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;


		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum][0];
		}


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Pooling_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Pooling_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Pooling_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 ch=0; ch<this->layerData.outputDataStruct.ch; ch++)
			{
				for(U32 outputZ=0; outputZ<this->layerData.outputDataStruct.z; outputZ++)
				{
					for(U32 outputY=0; outputY<this->layerData.outputDataStruct.y; outputY++)
					{
						for(U32 outputX=0; outputX<this->layerData.outputDataStruct.x; outputX++)
						{
							// �ő�l�𒲂ׂ�
							F32 maxValue = -FLT_MAX;
							for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
							{
								for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
								{
									for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
									{
										U32 inputX = outputX * this->layerData.layerStructure.FilterSize.x + filterX;
										U32 inputY = outputY * this->layerData.layerStructure.FilterSize.y + filterY;
										U32 inputZ = outputZ * this->layerData.layerStructure.FilterSize.z + filterZ;
										if(inputX >= this->layerData.inputDataStruct.x)
											continue;
										if(inputY >= this->layerData.inputDataStruct.y)
											continue;
										if(inputZ >= this->layerData.inputDataStruct.z)
											continue;

										U32 inputOffset = POSITION_TO_OFFSET_STRUCT(
																inputX,
																inputY,
																inputZ,
																ch,
																this->layerData.inputDataStruct);

										maxValue = max(maxValue, this->m_lppInputBuffer[batchNum][inputOffset]);
									}
								}
							}
							
							U32 outputOffset = POSITION_TO_OFFSET_STRUCT(outputX,outputY,outputZ,ch, this->layerData.outputDataStruct);
							this->lpOutputBuffer[batchNum][outputOffset] = maxValue;
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
	CONST_BATCH_BUFFER_POINTER Pooling_CPU::GetOutputBuffer()const
	{
		return &this->lppBatchOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode Pooling_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseOutputBuffer = this->GetOutputBuffer();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpOutputBuffer[batchNum], lppUseOutputBuffer[batchNum], sizeof(F32)*outputBufferCount);
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
	ErrorCode Pooling_CPU::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		this->m_lppDOutputBufferPrev = i_lppDOutputBufferPrev;
		
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			// �o�b�t�@���N���A
			memset(&this->lpDInputBuffer[batchNum][0], 0, this->inputBufferCount * sizeof(F32));
			
			for(U32 ch=0; ch<this->layerData.outputDataStruct.ch; ch++)
			{
				for(U32 outputZ=0; outputZ<this->layerData.outputDataStruct.z; outputZ++)
				{
					for(U32 outputY=0; outputY<this->layerData.outputDataStruct.y; outputY++)
					{
						for(U32 outputX=0; outputX<this->layerData.outputDataStruct.x; outputX++)
						{
							U32 outputOffset = POSITION_TO_OFFSET_STRUCT(outputX,outputY,outputZ,ch, this->layerData.outputDataStruct);

							// �ő�l�𒲂ׂ�
							for(S32 filterZ=0; filterZ<this->layerData.layerStructure.FilterSize.z; filterZ++)
							{
								for(S32 filterY=0; filterY<this->layerData.layerStructure.FilterSize.y; filterY++)
								{
									for(S32 filterX=0; filterX<this->layerData.layerStructure.FilterSize.x; filterX++)
									{
										U32 inputX = outputX * this->layerData.layerStructure.FilterSize.x + filterX;
										U32 inputY = outputY * this->layerData.layerStructure.FilterSize.y + filterY;
										U32 inputZ = outputZ * this->layerData.layerStructure.FilterSize.z + filterZ;
										if(inputX >= this->layerData.inputDataStruct.x)
											continue;
										if(inputY >= this->layerData.inputDataStruct.y)
											continue;
										if(inputZ >= this->layerData.inputDataStruct.z)
											continue;

										U32 inputOffset = POSITION_TO_OFFSET_STRUCT(
																inputX,
																inputY,
																inputZ,
																ch,
																this->layerData.inputDataStruct);

										if(this->m_lppInputBuffer[batchNum][inputOffset] == this->lpOutputBuffer[batchNum][outputOffset])
											this->lpDInputBuffer[batchNum][inputOffset] = this->m_lppDOutputBufferPrev[batchNum][outputOffset];
									}
								}
							}

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
	ErrorCode Pooling_CPU::ReflectionLearnError(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Pooling_CPU::GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode Pooling_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseDInputBuffer = this->GetDInputBuffer();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpDInputBuffer[batchNum], lppUseDInputBuffer[batchNum], sizeof(F32)*inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
