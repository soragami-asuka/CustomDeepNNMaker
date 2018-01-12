//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"MergeMax_DATA.hpp"
#include"MergeMax_FUNC.hpp"
#include"MergeMax_Base.h"

#include"MergeMax_CPU.h"
#include"MergeMax_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	MergeMax_CPU::MergeMax_CPU(Gravisbell::GUID guid, MergeMax_LayerData_CPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	MergeMax_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
	{
	}
	/** �f�X�g���N�^ */
	MergeMax_CPU::~MergeMax_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 MergeMax_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode MergeMax_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	MergeMax_LayerData_Base& MergeMax_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeMax_LayerData_Base& MergeMax_CPU::GetLayerData()const
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
	ErrorCode MergeMax_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());

		// ���͍����o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->lppBatchDInputBuffer.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->lppBatchDInputBuffer[inputNum].resize(this->GetBatchSize());
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MergeMax_CPU::PreProcessCalculate()
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
		this->lppBatchInputBuffer.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			this->lppBatchInputBuffer[inputNum].resize(this->GetBatchSize(), NULL);
		}

		// �o�̓o�b�t�@���쐬
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MergeMax_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode MergeMax_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->lppBatchInputBuffer[inputNum][batchNum] = &i_lppInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]];
		}
		// �o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];

		// �o�̓o�b�t�@��-FLT_MAX�ŃN���A
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
			{
				this->lppBatchOutputBuffer[batchNum][bufNum] = -FLT_MAX;
			}
		}

		// �ő�l���v�Z
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
			{
				U32 bufferSize = min(this->lpInputBufferCount[inputNum], outputBufferCount);

				for(U32 bufNum=0; bufNum<bufferSize; bufNum++)
				{
					this->lppBatchOutputBuffer[batchNum][bufNum] = max(this->lppBatchOutputBuffer[batchNum][bufNum], this->lppBatchInputBuffer[inputNum][batchNum][bufNum]);
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
	ErrorCode MergeMax_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];

		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					this->lppBatchDInputBuffer[inputNum][batchNum] = &o_lppDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]];
				}

				// �o�b�t�@�̃N���A
				memset(o_lppDInputBuffer[inputNum], 0, sizeof(F32)*this->lpInputBufferCount[inputNum]*this->GetBatchSize());
			}
			// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					this->lppBatchInputBuffer[inputNum][batchNum] = &i_lppInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]];
			}
			// �o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->lppBatchOutputBuffer[batchNum] = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);

			U32 CH_SIZE = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				U32 offset_output = 0;
				for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
				{
					U32 bufferSize = min(this->lpInputBufferCount[inputNum], outputBufferCount);

					for(U32 bufNum=0; bufNum<bufferSize; bufNum++)
					{
						if(this->lppBatchOutputBuffer[batchNum][bufNum] == this->lppBatchInputBuffer[inputNum][batchNum][bufNum])	// ���͂Əo�͂���v����ꍇ
						{
							this->lppBatchDInputBuffer[inputNum][batchNum][bufNum] = this->lppBatchDOutputBuffer[batchNum][bufNum];
						}
					}
				}
			}
		}

#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer0(this->GetBatchSize() * this->lpInputBufferCount[0]);
		memcpy(&lpTmpInputBuffer0[0], i_lppInputBuffer[0], sizeof(float)*lpTmpInputBuffer0.size());
		std::vector<float> lpTmpInputBuffer1(this->GetBatchSize() * this->lpInputBufferCount[1]);
		memcpy(&lpTmpInputBuffer1[0], i_lppInputBuffer[1], sizeof(float)*lpTmpInputBuffer1.size());

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size());

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size());

		std::vector<float> lpTmpDInputBuffer0(this->GetBatchSize() * this->lpInputBufferCount[0]);
		memcpy(&lpTmpDInputBuffer0[0], o_lppDInputBuffer[0], sizeof(float)*lpTmpDInputBuffer0.size());
		std::vector<float> lpTmpDInputBuffer1(this->GetBatchSize() * this->lpInputBufferCount[1]);
		memcpy(&lpTmpDInputBuffer1[0], o_lppDInputBuffer[1], sizeof(float)*lpTmpDInputBuffer1.size());
#endif


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode MergeMax_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
