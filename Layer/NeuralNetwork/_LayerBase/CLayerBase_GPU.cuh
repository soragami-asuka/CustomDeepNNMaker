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

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	//=================================
	// �P����� / �P��o��
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2SingleLayerBase_GPU : public Layer
	{
	protected:

		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** �R���X�g���N�^ */
		CNNSingle2SingleLayerBase_GPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2SingleLayerBase_GPU()
		{
		}

	public:
		/** ���o�̓o�b�t�@�̊m�ۂƈꎞ�o�b�t�@�̗\�� */
		ErrorCode ReserveMemory()
		{
			// �ꎞ�o�b�t�@�̃T�C�Y��o�^
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"input[0]",   sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"output[0]",  sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"dinput[0]",  sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());

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
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpy(lppInputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// �o�̓o�b�t�@���m��
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");

			Gravisbell::ErrorCode err = this->Calculate_device(lppInputBuffer, lppOutputBuffer);

			// �o�̓o�b�t�@���z�X�g�ɃR�s�[
			cudaMemcpy(o_lppOutputBuffer, lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);

			// �o�b�t�@���J��
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");

			return err;
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
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, i_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���͌덷�o�b�t�@���m��
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// ����
			cudaThreadSynchronize();

			// ���Z
			ErrorCode err = CalculateDInput_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// ���͌덷�o�b�t�@���z�X�g�ɃR�s�[
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// �o�b�t�@���
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, i_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���͌덷�o�b�t�@���m��
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// ����
			cudaThreadSynchronize();

			// ���Z
			ErrorCode err = Training_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// ���͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// �o�b�t�@���
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}
	};

	//=================================
	// �P����� / �����o��
	//=================================
	template<class Layer, class LayerData>
	class CNNSingle2MultLayerBase_GPU : public Layer
	{
	protected:
		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;
		
		std::vector<BATCH_BUFFER_POINTER>	lppDOutputBuffer_d;	/**< �o�͌덷�o�b�t�@�̔z��(�f�o�C�X������). <�o�͐惌�C���[��> */
		std::vector<std::wstring>		lpDOutputBufferID;		/**< �o�͌덷�o�b�t�@�ɕt����ꂽID */

	public:
		/** �R���X�g���N�^ */
		CNNSingle2MultLayerBase_GPU(Gravisbell::GUID guid, LayerData& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2MultLayerBase_GPU()
		{
		}

	public:
		/** ���o�̓o�b�t�@�̊m�ۂƈꎞ�o�b�t�@�̗\�� */
		ErrorCode ReserveMemory()
		{
			// ID���X�g��������
			this->lpDOutputBufferID.clear();

			// �ꎞ�o�b�t�@�̃T�C�Y��o�^
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"input[0]",   sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"output[0]",  sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"dinput[0]",  sizeof(F32) * this->GetInputBufferCount()  * this->GetBatchSize());
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
			{
				wchar_t szID[256];
				swprintf(szID, sizeof(szID)-1, L"doutput[%d]", outputNo);

				this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), szID, sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
				this->lpDOutputBufferID.push_back(szID);
			}

			// �o�͌덷�o�b�t�@�z����m��
			this->lppDOutputBuffer_d.resize(this->GetOutputToLayerCount());

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
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpy(lppInputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// �o�̓o�b�t�@���m��
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");

			Gravisbell::ErrorCode err = this->Calculate_device(lppInputBuffer, lppOutputBuffer);

			// �o�̓o�b�t�@���z�X�g�ɃR�s�[
			cudaMemcpy(o_lppOutputBuffer, lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);

			// �o�b�t�@���J��
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");

			return err;
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
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			// �o�͌덷�o�b�t�@�z����쐬
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
			{
				// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
				this->lppDOutputBuffer_d[outputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

				// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
				cudaMemcpyAsync(this->lppDOutputBuffer_d[outputNo], i_lppDOutputBuffer[outputNo], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, i_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���͌덷�o�b�t�@���m��
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// ����
			cudaThreadSynchronize();

			// ���Z
			ErrorCode err = CalculateDInput_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, (const float**)&this->lppDOutputBuffer_d[0]);


			// ���͌덷�o�b�t�@���z�X�g�ɃR�s�[
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// �o�b�t�@���
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

			return err;
		}

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[])
		{
			// �o�͌덷�o�b�t�@�z����쐬
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
			{
				// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
				this->lppDOutputBuffer_d[outputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

				// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
				cudaMemcpyAsync(this->lppDOutputBuffer_d[outputNo], i_lppDOutputBuffer[outputNo], sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, i_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
			cudaMemcpyAsync(lppInputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���͌덷�o�b�t�@���m��
			F32* lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
				lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

			// ����
			cudaThreadSynchronize();

			// ���Z
			ErrorCode err = Training_device(lppInputBuffer, lppDInputBuffer, lppOutputBuffer, (const float**)&this->lppDOutputBuffer_d[0]);


			// ���͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			if(o_lppDInputBuffer)
			{
				cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
			}

			// �o�b�t�@���
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			for(U32 outputNo=0; outputNo<this->GetOutputToLayerCount(); outputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDOutputBufferID[outputNo].c_str());

			return err;
		}
	};

	//=================================
	// �������� / �P��o��
	//=================================
	template<class Layer, class LayerData>
	class CNNMult2SingleLayerBase_GPU : public Layer
	{
	protected:
		std::vector<std::wstring>		lpInputBufferID;		/**< ���̓o�b�t�@�ɕt����ꂽID */
		std::vector<std::wstring>		lpDInputBufferID;		/**< ���͌덷�o�b�t�@�ɕt����ꂽID */

		std::vector<BATCH_BUFFER_POINTER>	lppInputBuffer_d;	/**< ���̓o�b�t�@�̔z��(�f�o�C�X������). <���͌����C���[��> */
		std::vector<BATCH_BUFFER_POINTER>	lppDInputBuffer_d;	/**< ���͌덷�o�b�t�@�̔z��(�f�o�C�X������). <���͌����C���[��> */


		Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;

	public:
		/** �R���X�g���N�^ */
		CNNMult2SingleLayerBase_GPU(Gravisbell::GUID guid, LayerData& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
			:	Layer							(guid, i_layerData, i_lpInputDataStruct, i_temporaryMemoryManager)
			,	temporaryMemoryManager			(i_temporaryMemoryManager)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNMult2SingleLayerBase_GPU()
		{
		}

	public:
		/** ���o�̓o�b�t�@�̊m�ۂƈꎞ�o�b�t�@�̗\�� */
		ErrorCode ReserveMemory()
		{
			this->lpInputBufferID.clear();
			this->lpDInputBufferID.clear();

			// �ꎞ�o�b�t�@�̃T�C�Y��o�^
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				wchar_t szID[256];

				swprintf(szID, sizeof(szID)-1, L"input[%d]", inputNo);
				this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), szID,  sizeof(F32) * this->GetInputBufferCount(inputNo) * this->GetBatchSize());
				this->lpInputBufferID.push_back(szID);

				swprintf(szID, sizeof(szID)-1, L"dinput[%d]", inputNo);
				this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), szID,  sizeof(F32) * this->GetInputBufferCount(inputNo) * this->GetBatchSize());
				this->lpDInputBufferID.push_back(szID);
			}
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"output[0]",  sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());
			this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32) * this->GetOutputBufferCount() * this->GetBatchSize());

			// ����/���͌덷�o�b�t�@�̃A�h���X�z����쐬
			this->lppInputBuffer_d.resize(this->GetInputDataCount());
			this->lppDInputBuffer_d.resize(this->GetInputDataCount());

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
		ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
		{
			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				this->lppInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
				cudaMemcpy(this->lppInputBuffer_d[inputNo], i_lppInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// �o�̓o�b�t�@���m��
			F32* lppOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");

			Gravisbell::ErrorCode err = this->Calculate_device((const F32**)&this->lppInputBuffer_d[0], lppOutputBuffer);

			// �o�̓o�b�t�@���z�X�g�ɃR�s�[
			cudaMemcpy(o_lppOutputBuffer, lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);

			// �o�b�t�@���J��
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");

			return err;
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
		ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, i_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				this->lppInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
				cudaMemcpyAsync(this->lppInputBuffer_d[inputNo], i_lppInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// ���͌덷�o�b�t�@���m��
			F32** lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					this->lppDInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
				lppDInputBuffer = &this->lppDInputBuffer_d[0];
			}

			// ����
			cudaThreadSynchronize();

			// ���Z
			ErrorCode err = CalculateDInput_device((const F32**)&this->lppInputBuffer_d[0], lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// ���͌덷�o�b�t�@���z�X�g�ɃR�s�[
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					cudaMemcpy(o_lppDInputBuffer[inputNo], lppDInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyDeviceToHost);
					this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
			}

			// �o�b�t�@���
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
		{
			// �o�͌덷�o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
			cudaMemcpyAsync(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
			F32* lppOutputBuffer =  (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"output[0]");
			cudaMemcpyAsync(lppOutputBuffer, i_lppOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

			// ���̓o�b�t�@���f�o�C�X�ɃR�s�[
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
			{
				this->lppInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
				cudaMemcpyAsync(this->lppInputBuffer_d[inputNo], i_lppInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyHostToDevice);
			}

			// ���͌덷�o�b�t�@���m��
			F32** lppDInputBuffer = NULL;
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					this->lppDInputBuffer_d[inputNo] = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
				lppDInputBuffer = &this->lppDInputBuffer_d[0];
			}

			// ����
			cudaThreadSynchronize();


			// ���Z
			ErrorCode err = Training_device((const F32**)&this->lppInputBuffer_d[0], lppDInputBuffer, lppOutputBuffer, lppDOutputBuffer);


			// ���͌덷�o�b�t�@���z�X�g�ɃR�s�[
			if(o_lppDInputBuffer)
			{
				for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				{
					cudaMemcpy(o_lppDInputBuffer[inputNo], lppDInputBuffer[inputNo], sizeof(F32)*this->GetInputBufferCount(inputNo)*this->GetBatchSize(), cudaMemcpyDeviceToHost);
					this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpDInputBufferID[inputNo].c_str());
				}
			}

			// �o�b�t�@���
			for(U32 inputNo=0; inputNo<this->GetInputDataCount(); inputNo++)
				this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), this->lpInputBufferID[inputNo].c_str());
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"output[0]");
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

			return err;
		}
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
