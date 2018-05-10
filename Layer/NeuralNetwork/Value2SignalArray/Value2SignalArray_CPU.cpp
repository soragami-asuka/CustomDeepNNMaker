//======================================
// �o�͐M���������C���[
// CPU�����p
//======================================
#include"stdafx.h"

#include<algorithm>

#include"Value2SignalArray_DATA.hpp"
#include"Value2SignalArray_FUNC.hpp"
#include"Value2SignalArray_Base.h"

#include"Value2SignalArray_CPU.h"
#include"Value2SignalArray_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Value2SignalArray_CPU::Value2SignalArray_CPU(Gravisbell::GUID guid, Value2SignalArray_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Value2SignalArray_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
	{
	}
	/** �f�X�g���N�^ */
	Value2SignalArray_CPU::~Value2SignalArray_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Value2SignalArray_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Value2SignalArray_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Value2SignalArray_LayerData_Base& Value2SignalArray_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Value2SignalArray_LayerData_Base& Value2SignalArray_CPU::GetLayerData()const
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
	ErrorCode Value2SignalArray_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ���͌덷/�o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Value2SignalArray_CPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �`�����l���̃o�b�t�@�T�C�Y��ۑ�
		this->channelSize = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		// ����/�o�̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->lppBatchInputBuffer.resize(this->GetBatchSize(), NULL);
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Value2SignalArray_CPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Value2SignalArray_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// �o�̓o�b�t�@��0����
		memset(o_lppOutputBuffer, 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 z=0; z<this->GetInputDataStruct().z; z++)
			{
				for(U32 y=0; y<this->GetInputDataStruct().y; y++)
				{
					for(U32 x=0; x<this->GetInputDataStruct().x; x++)
					{
						for(U32 inputCh=0; inputCh<this->GetInputDataStruct().ch; inputCh++)
						{
							U32 inputOffset = this->GetInputDataStruct().POSITION_TO_OFFSET(x, y, z, inputCh);

							F32 value = this->lppBatchInputBuffer[batchNum][inputOffset];

							// �o�̓`�����l���ԍ���float�Ōv�Z
							F32 fOutputCh = (value - this->layerData.layerStructure.inputMinValue) / (this->layerData.layerStructure.inputMaxValue - this->layerData.layerStructure.inputMinValue) * this->layerData.layerStructure.resolution;
							fOutputCh = std::min<F32>((F32)this->layerData.layerStructure.resolution-1, std::max<F32>(0, fOutputCh));

							// �o�̓`�����l���ԍ��𐮐��ƒ[���ɕ���
							U32 iOutputCh = (U32)fOutputCh;
							F32 t = fOutputCh - iOutputCh;

							U32 outputOffset0 = this->GetOutputDataStruct().POSITION_TO_OFFSET(x,y,z, iOutputCh + inputCh*this->layerData.layerStructure.resolution);
							this->lppBatchOutputBuffer[batchNum][outputOffset0] = (1.0f - t);
							if((S32)iOutputCh+1 < this->layerData.layerStructure.resolution)
							{
								U32 outputOffset1 = this->GetOutputDataStruct().POSITION_TO_OFFSET(x,y,z, (iOutputCh + 1) + inputCh*this->layerData.layerStructure.resolution);
								this->lppBatchOutputBuffer[batchNum][outputOffset1] = t;
							}
						}
					}
				}
			}

		}

#if _DEBUG
		std::vector<F32> lpInputBuffer(this->inputBufferCount * this->GetBatchSize());
		memcpy(&lpInputBuffer[0], i_lppInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize());

		std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
		memcpy(&lpOutputBuffer[0], o_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize());
#endif

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
	ErrorCode Value2SignalArray_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			// ���o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
				this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
				this->lppBatchOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
			}

			std::vector<F32> lpTmpTeachBuffer(this->layerData.layerStructure.resolution);

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 z=0; z<this->GetInputDataStruct().z; z++)
				{
					for(U32 y=0; y<this->GetInputDataStruct().y; y++)
					{
						for(U32 x=0; x<this->GetInputDataStruct().x; x++)
						{
							for(U32 inputCh=0; inputCh<this->GetInputDataStruct().ch; inputCh++)
							{
								F32 teachValue = 0.0f;

								// signal�̋��t�M����䗦�Ƃ��ẮA�ŏ��l�A�ő�l���|���Z����value�̋��t�M�����쐬����
								for(S32 i=0; i<this->layerData.layerStructure.resolution; i++)
								{
									F32 t = (this->layerData.layerStructure.inputMaxValue - this->layerData.layerStructure.inputMinValue) * i / this->layerData.layerStructure.resolution - this->layerData.layerStructure.inputMinValue;

									U32 outputCh = inputCh * this->layerData.layerStructure.resolution + i;
									U32 outputOffset = this->GetOutputDataStruct().POSITION_TO_OFFSET(x,y,z,outputCh);

									F32 teachSignal = this->lppBatchOutputBuffer[batchNum][outputOffset] + this->lppBatchDOutputBuffer[batchNum][outputOffset];

									teachValue += t * teachSignal;
								}

								// ���t�M��-���͐M�������͍���
								U32 inputOffset = this->GetInputDataStruct().POSITION_TO_OFFSET(x,y,z,inputCh);

								this->lppBatchDInputBuffer[batchNum][inputOffset] = teachValue - this->lppBatchInputBuffer[batchNum][inputOffset];
							}
						}
					}
				}
#if _DEBUG
				std::vector<F32> lpDOutputBuffer(this->outputBufferCount);
				memcpy(&lpDOutputBuffer[0], this->lppBatchDOutputBuffer[batchNum], sizeof(F32) * this->outputBufferCount);

				std::vector<F32> lpDInputBuffer(this->inputBufferCount);
				memcpy(&lpDInputBuffer[0], this->lppBatchDInputBuffer[batchNum], sizeof(F32) * this->inputBufferCount);
#endif
			}

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Value2SignalArray_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
