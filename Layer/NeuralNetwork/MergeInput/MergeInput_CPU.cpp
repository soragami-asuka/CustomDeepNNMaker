//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"MergeInput_DATA.hpp"
#include"MergeInput_FUNC.hpp"
#include"MergeInput_Base.h"

#include"MergeInput_CPU.h"
#include"MergeInput_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	MergeInput_CPU::MergeInput_CPU(Gravisbell::GUID guid, MergeInput_LayerData_CPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct)
		:	MergeInput_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
	{
	}
	/** �f�X�g���N�^ */
	MergeInput_CPU::~MergeInput_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 MergeInput_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode MergeInput_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	MergeInput_LayerData_Base& MergeInput_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeInput_LayerData_Base& MergeInput_CPU::GetLayerData()const
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
	ErrorCode MergeInput_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDOutputBufferPrev.resize(this->GetBatchSize());

		// ���͍����o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDInputBuffer.resize(this->GetInputDataCount());
		this->m_lppBatchDInputBuffer.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->m_lppBatchDInputBuffer[inputNum].resize(this->GetBatchSize());
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MergeInput_CPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->lpInputBufferCount.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->lpInputBufferCount[inputNum] = this->GetInputBufferCount(inputNum);
		}

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ���̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lppInputBuffer.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->m_lppInputBuffer[inputNum].resize(this->GetBatchSize(), NULL);
		}

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
	ErrorCode MergeInput_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode MergeInput_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer[])
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->m_lppInputBuffer[inputNum][batchNum] = &i_lpInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]];
		}

		switch(this->layerData.layerStructure.mergeDirection)
		{
		case MergeInput::LayerStructure::mergeDirection_ch:
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					U32 offset = 0;
					for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
					{
						memcpy(
							&this->lppBatchOutputBuffer[batchNum][offset],
							this->m_lppInputBuffer[inputNum][batchNum],
							sizeof(F32) * this->lpInputBufferCount[inputNum]);

						offset += this->lpInputBufferCount[inputNum];
					}
				}
			}
			break;
		case MergeInput::LayerStructure::mergeDirection_x:
			break;
		case MergeInput::LayerStructure::mergeDirection_y:
			break;
		case MergeInput::LayerStructure::mergeDirection_z:
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER MergeInput_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode MergeInput_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode MergeInput_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppDOutputBufferPrev[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];

		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				this->m_lppDInputBuffer[inputNum] = o_lppDInputBuffer[inputNum];
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					this->m_lppBatchDInputBuffer[inputNum][batchNum] = &this->m_lppDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]];
				}
			}

			switch(this->layerData.layerStructure.mergeDirection)
			{
			case MergeInput::LayerStructure::mergeDirection_ch:
				{
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						U32 offset = 0;
						for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
						{
							memcpy(
								this->m_lppBatchDInputBuffer[inputNum][batchNum],
								&this->m_lppDOutputBufferPrev[batchNum][offset],
								sizeof(F32) * this->lpInputBufferCount[inputNum]);

							offset += this->lpInputBufferCount[inputNum];
						}
					}
				}
				break;
			case MergeInput::LayerStructure::mergeDirection_x:
				break;
			case MergeInput::LayerStructure::mergeDirection_y:
				break;
			case MergeInput::LayerStructure::mergeDirection_z:
				break;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode MergeInput_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER MergeInput_CPU::GetDInputBuffer(U32 i_dataNum)const
	{
		if(i_dataNum >= this->m_lppDInputBuffer.size())
			return NULL;

		return this->m_lppDInputBuffer[i_dataNum];
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode MergeInput_CPU::GetDInputBuffer(U32 i_dataNum, BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount(i_dataNum);

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(i_dataNum), sizeof(F32)*inputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
