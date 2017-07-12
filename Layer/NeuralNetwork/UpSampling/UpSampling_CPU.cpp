//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌������C���[
// CPU�����p
//======================================
#include"stdafx.h"

#include"UpSampling_DATA.hpp"
#include"UpSampling_FUNC.hpp"
#include"UpSampling_Base.h"

#include"UpSampling_CPU.h"
#include"UpSampling_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define POSITION_TO_OFFSET(x,y,z,ch,xSize,ySize,zSize,chSize)		((((((ch)*(zSize)+(z))*(ySize))+(y))*(xSize))+(x))
#define POSITION_TO_OFFSET_STRUCT(inX,inY,inZ,inCh,structure)		POSITION_TO_OFFSET(inX, inY, inZ, inCh, structure.x, structure.y, structure.z, structure.ch)
#define POSITION_TO_OFFSET_VECTOR(inX,inY,inZ,inCh,vector,chSize)	POSITION_TO_OFFSET(inX, inY, inZ, inCh, vector.x,    vector.y,    vector.z,    chSize)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	UpSampling_CPU::UpSampling_CPU(Gravisbell::GUID guid, UpSampling_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct)
		:	UpSampling_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
	{
	}
	/** �f�X�g���N�^ */
	UpSampling_CPU::~UpSampling_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 UpSampling_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode UpSampling_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	UpSampling_LayerData_Base& UpSampling_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const UpSampling_LayerData_Base& UpSampling_CPU::GetLayerData()const
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
	ErrorCode UpSampling_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDOutputBuffer.resize(this->GetBatchSize());
		// ���͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDInputBuffer.resize(this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpSampling_CPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

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
	ErrorCode UpSampling_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode UpSampling_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		// �o�̓o�b�t�@���N���A
		memset(&this->lpOutputBuffer[0], 0, sizeof(F32)*this->lpOutputBuffer.size());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
			{
				for(U32 inputZ=0; inputZ<this->GetInputDataStruct().z; inputZ++)
				{
					for(U32 inputY=0; inputY<this->GetInputDataStruct().y; inputY++)
					{
						for(U32 inputX=0; inputX<this->GetInputDataStruct().x; inputX++)
						{
							U32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, ch, this->GetInputDataStruct());

							switch(this->layerData.layerStructure.PaddingType)
							{
							case UpSampling::LayerStructure::PaddingType_value:
								{
									for(S32 offsetZ=0; offsetZ<this->layerData.layerStructure.UpScale.z; offsetZ++)
									{
										for(S32 offsetY=0; offsetY<this->layerData.layerStructure.UpScale.y; offsetY++)
										{
											for(S32 offsetX=0; offsetX<this->layerData.layerStructure.UpScale.x; offsetX++)
											{
												U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
													inputX*this->layerData.layerStructure.UpScale.x + offsetX,
													inputY*this->layerData.layerStructure.UpScale.y + offsetY,
													inputZ*this->layerData.layerStructure.UpScale.z + offsetZ,
													ch,
													this->GetOutputDataStruct());

												this->lppBatchOutputBuffer[batchNum][outputOffset] = this->m_lppInputBuffer[batchNum][inputOffset];
											}
										}
									}
								}
								break;
							case UpSampling::LayerStructure::PaddingType_zero:
								{
									U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
										inputX*this->layerData.layerStructure.UpScale.x + 0,
										inputY*this->layerData.layerStructure.UpScale.y + 0,
										inputZ*this->layerData.layerStructure.UpScale.z + 0,
										ch,
										this->GetOutputDataStruct());

									this->lppBatchOutputBuffer[batchNum][outputOffset] = this->m_lppInputBuffer[batchNum][inputOffset];
								}
								break;
							}
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
	CONST_BATCH_BUFFER_POINTER UpSampling_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode UpSampling_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode UpSampling_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];

		// ���͌덷�v�Z
		this->m_lpDInputBuffer = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];

			// ���͌덷�o�b�t�@��������
			memset(this->m_lpDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

		
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 ch=0; ch<this->GetInputDataStruct().ch; ch++)
				{
					for(U32 inputZ=0; inputZ<this->GetInputDataStruct().z; inputZ++)
					{
						for(U32 inputY=0; inputY<this->GetInputDataStruct().y; inputY++)
						{
							for(U32 inputX=0; inputX<this->GetInputDataStruct().x; inputX++)
							{
								U32 inputOffset = POSITION_TO_OFFSET_STRUCT(inputX, inputY, inputZ, ch, this->GetInputDataStruct());

								switch(this->layerData.layerStructure.PaddingType)
								{
								case UpSampling::LayerStructure::PaddingType_value:
									{
										for(S32 offsetZ=0; offsetZ<this->layerData.layerStructure.UpScale.z; offsetZ++)
										{
											for(S32 offsetY=0; offsetY<this->layerData.layerStructure.UpScale.y; offsetY++)
											{
												for(S32 offsetX=0; offsetX<this->layerData.layerStructure.UpScale.x; offsetX++)
												{
													U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
														inputX*this->layerData.layerStructure.UpScale.x + offsetX,
														inputY*this->layerData.layerStructure.UpScale.y + offsetY,
														inputZ*this->layerData.layerStructure.UpScale.z + offsetZ,
														ch,
														this->GetOutputDataStruct());


													this->m_lppDInputBuffer[batchNum][inputOffset] += this->m_lppDOutputBuffer[batchNum][outputOffset];
												}
											}
										}
									}
									break;
								case UpSampling::LayerStructure::PaddingType_zero:
									{
										U32 outputOffset = POSITION_TO_OFFSET_STRUCT(
											inputX*this->layerData.layerStructure.UpScale.x + 0,
											inputY*this->layerData.layerStructure.UpScale.y + 0,
											inputZ*this->layerData.layerStructure.UpScale.z + 0,
											ch,
											this->GetOutputDataStruct());

										this->m_lppDInputBuffer[batchNum][inputOffset] = this->m_lppDOutputBuffer[batchNum][outputOffset];
									}
									break;
								}

							}
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
	ErrorCode UpSampling_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER UpSampling_CPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode UpSampling_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
