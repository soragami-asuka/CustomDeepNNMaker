//======================================
// �����֐����C���[
// GPU�����p
//======================================
#ifndef __BatchExponentialNormalization_GPU_H__
#define __BatchExponentialNormalization_GPU_H__

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"BatchExponentialNormalization_DATA.hpp"
#include"BatchExponentialNormalization_FUNC.hpp"
#include"BatchExponentialNormalization_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class BatchExponentialNormalization_GPU : public BatchExponentialNormalization_Base
{
private:
	// �f�[�^�{��
	class BatchExponentialNormalization_LayerData_GPU& layerData;
//	BatchExponentialNormalization::LearnDataStructure learnData;

	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */
	U32 channeclBufferCount;				/**< 1�`�����l��������̃o�b�t�@�� */

	// �w�K�p�̃f�[�^
	U32 learnCount;		/**< �w�K���s�� */
	thrust::device_vector<F32> lpTmpMean;			/**< ���ϒl�i�[�p�̈ꎞ�ϐ� */
	thrust::device_vector<F32> lpTmpVariance;		/**< ���U�l�i�[�p�̈ꎞ�ϐ� */

	thrust::device_vector<F32> lpLearnMean;			/**< �w�K�p���ϒl�i�[�p�̈ꎞ�ϐ� */
	thrust::device_vector<F32> lpLearnVariance;		/**< �w�K�p���U�l�i�[�p�̈ꎞ�ϐ� */

	// ���Z�����p�̃o�b�t�@
	cudnnHandle_t cudnnHandle;

	// CUDNN�p�f�[�^�\����`
	cudnnTensorDescriptor_t			inputTensorDesc;			/**< ���̓f�[�^�\�� */
	cudnnTensorDescriptor_t			outputTensorDesc;			/**< �o�̓f�[�^�\�� */
	cudnnTensorDescriptor_t			paramTensorDesc;			/**< �X�P�[��,�o�C�A�X,���ς̊e�l�̃f�[�^�\�� */

	thrust::device_vector<F32> lpDBias;		/**< �o�C�A�X�̕ω��� */
	thrust::device_vector<F32> lpDScale;		/**< �X�P�[���̕ω��� */

	Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;	/**< �ꎞ�o�b�t�@�Ǘ� */

public:
	/** �R���X�g���N�^ */
	BatchExponentialNormalization_GPU(Gravisbell::GUID guid, class BatchExponentialNormalization_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** �f�X�g���N�^ */
	virtual ~BatchExponentialNormalization_GPU();

public:
	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 GetLayerKind()const;

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Initialize(void);


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	BatchExponentialNormalization_LayerData_Base& GetLayerData();
	const BatchExponentialNormalization_LayerData_Base& GetLayerData()const;


public:
	//================================
	// ���O����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn();

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate();


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLoop();



public:
	//================================
	// ���Z����
	//================================
	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);

public:
	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif