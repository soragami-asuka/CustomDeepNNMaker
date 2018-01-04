//============================================
// �S�Ẵj���[�����l�b�g���[�N�n���C���[�̃x�[�X�ƂȂ鋤�ʏ���
// ���O�Ŏ����\�ł���ΕK���p������K�v�͂Ȃ�
//============================================
#ifndef __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_GPU_H__
#define __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_GPU_H__


#include"CLayerBase.h"

#include<Common/Common.h>
#include<Common/ErrorCode.h>
#include<Common/ITemporaryMemoryManager.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	//=================================
	// �P����� / �P��o��
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2SingleLayerBase_CPU : public Layer
	{
	protected:
		// ���o�̓o�b�t�@
		std::vector<F32>				lpInputBuffer_h;		/**< ���̓o�b�t�@ <�o�b�`��><���͐M����> */
		std::vector<F32>				lpOutputBuffer_h;		/**< �o�̓o�b�t�@ <�o�b�`��><�o�͐M����> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** �R���X�g���N�^ */
		CNNSingle2SingleLayerBase_CPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2SingleLayerBase_CPU()
		{
		}

	public:
		/** ���o�̓o�b�t�@�̊m�ۂƈꎞ�o�b�t�@�̗\�� */
		ErrorCode ReserveMemory()
		{
			// �o�̓o�b�t�@���m��
			this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		using Layer::Calculate;
		using Layer::CalculateDInput;
		using Layer::Training;

		//================================
		// ���O���Z
		//================================
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2SingleLayer>::PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2SingleLayer>::PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}

		//================================
		// ���Z����
		//================================
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
		{
			// ���̓o�b�t�@���R�s�[
			if(this->lpInputBuffer_h.empty())
				this->lpInputBuffer_h.resize(this->GetInputBufferCount() * this->GetBatchSize());
			memcpy(&this->lpInputBuffer_h[0], i_lpInputBuffer, sizeof(F32)*this->lpInputBuffer_h.size());

			return this->Calculate_device(&this->lpInputBuffer_h[0], &this->lpOutputBuffer_h[0]);
		}
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// ���Z
			ErrorCode err = Layer::Calculate_device(i_lpInputBuffer, o_lppOutputBuffer);

			// �o�̓o�b�t�@�����C���[���������ɃR�s�[
			memcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->lpOutputBuffer_h.size());

			return err;
		}


		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpOutputBuffer_h[0];
		}
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

			return ErrorCode::ERROR_CODE_NONE;
		}


	public:
		//================================
		// �w�K����
		//================================
		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  CalculateDInput_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  Training_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}
	};


	//=================================
	// �P����� / �����o��
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2MultLayerBase_CPU : public Layer
	{
	protected:
		// ���o�̓o�b�t�@
		std::vector<F32>				lpInputBuffer_h;		/**< ���̓o�b�t�@ <�o�b�`��><���͐M����> */
		std::vector<F32>				lpOutputBuffer_h;		/**< �o�̓o�b�t�@ <�o�b�`��><�o�͐M����> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** �R���X�g���N�^ */
		CNNSingle2MultLayerBase_CPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2MultLayerBase_CPU()
		{
		}

	public:
		/** ���o�̓o�b�t�@�̊m�ۂƈꎞ�o�b�t�@�̗\�� */
		ErrorCode ReserveMemory()
		{
			// �o�̓o�b�t�@���m��
			this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		using Layer::Calculate;
		using Layer::CalculateDInput;
		using Layer::Training;

		//================================
		// ���O���Z
		//================================
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2MultLayer>::PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNSingle2MultLayer>::PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}

		//================================
		// ���Z����
		//================================
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
		{
			// ���̓o�b�t�@���R�s�[
			if(this->lpInputBuffer_h.empty())
				this->lpInputBuffer_h.resize(this->GetInputBufferCount() * this->GetBatchSize());
			memcpy(&this->lpInputBuffer_h[0], i_lpInputBuffer, sizeof(F32)*this->lpInputBuffer_h.size());

			return this->Calculate_device(&this->lpInputBuffer_h[0], &this->lpOutputBuffer_h[0]);
		}
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// ���Z
			ErrorCode err = Layer::Calculate_device(i_lpInputBuffer, o_lppOutputBuffer);

			// �o�̓o�b�t�@�����C���[���������ɃR�s�[
			memcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->lpOutputBuffer_h.size());

			return err;
		}


		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpOutputBuffer_h[0];
		}
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

			return ErrorCode::ERROR_CODE_NONE;
		}


	public:
		//================================
		// �w�K����
		//================================
		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz��
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			return  CalculateDInput_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			return  Training_device(&this->lpInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}
	};


	//=================================
	// �������� / �P��o��
	//=================================
	template<class Layer, class LayerData>
	class CNNMult2SingleLayerBase_CPU : public Layer
	{
	protected:
		// ���o�̓o�b�t�@
		std::vector<std::vector<F32>>	lppInputBuffer_h;		/**< ���̓o�b�t�@ <���̓��C���[��><�o�b�`��*���͐M����> */
		std::vector<F32>				lpOutputBuffer_h;		/**< �o�̓o�b�t�@ <�o�b�`��><�o�͐M����> */

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** �R���X�g���N�^ */
		CNNMult2SingleLayerBase_CPU(Gravisbell::GUID guid, LayerData& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_lpInputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNMult2SingleLayerBase_CPU()
		{
		}

	public:
		/** ���o�̓o�b�t�@�̊m�ۂƈꎞ�o�b�t�@�̗\�� */
		ErrorCode ReserveMemory()
		{
			// �o�̓o�b�t�@���m��
			this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		using Layer::Calculate;
		using Layer::CalculateDInput;
		using Layer::Training;

		//================================
		// ���O���Z
		//================================
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNMult2SingleLayer>::PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			ErrorCode err = CLayerBase<INNMult2SingleLayer>::PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			return this->ReserveMemory();
		}

		//================================
		// ���Z����
		//================================
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[])
		{
			// ���̓o�b�t�@���R�s�[
			if(this->lppInputBuffer_h.empty())
			{
				this->lppInputBuffer_h.resize(this->GetInputDataCount());

				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
					this->lppInputBuffer_h[inputNo].resize(this->GetInputBufferCount(inputNo) * this->GetBatchSize());
			}
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				memcpy(&this->lppInputBuffer_h[inputNo][0], i_lppInputBuffer[inputNo], sizeof(F32)*this->lppInputBuffer_h[inputNo].size());

			return this->Calculate_device((const F32**)&this->lppInputBuffer_h[0], &this->lpOutputBuffer_h[0]);
		}
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// ���Z
			ErrorCode err = Layer::Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);

			// �o�̓o�b�t�@�����C���[���������ɃR�s�[
			memcpy(&this->lpOutputBuffer_h[0], o_lppOutputBuffer, sizeof(F32)*this->lpOutputBuffer_h.size());

			return err;
		}


		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpOutputBuffer_h[0];
		}
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

			return ErrorCode::ERROR_CODE_NONE;
		}


	public:
		//================================
		// �w�K����
		//================================
		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  CalculateDInput_device((const F32**)&this->lppInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			return  Training_device((const F32**)&this->lppInputBuffer_h[0], o_lppDInputBuffer, &this->lpOutputBuffer_h[0], i_lppDOutputBuffer);
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
