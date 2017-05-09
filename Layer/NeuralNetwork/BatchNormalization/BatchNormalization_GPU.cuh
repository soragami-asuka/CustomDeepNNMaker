//======================================
// �����֐����C���[
// GPU�����p
//======================================
#ifndef __BatchNormalization_GPU_H__
#define __BatchNormalization_GPU_H__

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

#include"BatchNormalization_DATA.hpp"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class BatchNormalization_GPU : public BatchNormalization_Base
{
private:
	// �f�[�^�{��
	class BatchNormalization_LayerData_GPU& layerData;
	BatchNormalization::LearnDataStructure learnData;

	// ���o�̓o�b�t�@
	thrust::device_vector<F32>			lpOutputBuffer;		/**< �o�̓o�b�t�@ <�o�b�`��><���͐M����> */
	thrust::device_vector<F32>			lpDInputBuffer;		/**< ���͌덷���� <�o�b�`��><���͐M����> */

	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */
	U32 channeclBufferCount;				/**< 1�`�����l��������̃o�b�t�@�� */

	// �w�K�p�̃f�[�^
	bool onLearnMode;	/**< �w�K�������t���O */
	U32 learnCount;		/**< �w�K���s�� */
	thrust::device_vector<F32> lpTmpMean;			/**< ���ϒl�i�[�p�̈ꎞ�ϐ� */
	thrust::device_vector<F32> lpTmpVariance;		/**< ���U�l�i�[�p�̈ꎞ�ϐ� */

	// ���Z���̓��̓f�[�^
	CONST_BATCH_BUFFER_POINTER m_lppInputBuffer;		/**< ���Z���̓��̓f�[�^ */
	CONST_BATCH_BUFFER_POINTER m_lppDOutputBufferPrev;	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */

	// ���Z�����p�̃o�b�t�@
	cudnnHandle_t cudnnHandle;

	// CUDNN�p�f�[�^�\����`
	cudnnTensorDescriptor_t			inputTensorDesc;			/**< ���̓f�[�^�\�� */
	cudnnTensorDescriptor_t			outputTensorDesc;			/**< �o�̓f�[�^�\�� */
	cudnnTensorDescriptor_t			paramTensorDesc;			/**< �X�P�[��,�o�C�A�X,���ς̊e�l�̃f�[�^�\�� */

public:
	/** �R���X�g���N�^ */
	BatchNormalization_GPU(Gravisbell::GUID guid, class BatchNormalization_LayerData_GPU& i_layerData);
	/** �f�X�g���N�^ */
	virtual ~BatchNormalization_GPU();

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
	BatchNormalization_LayerData_Base& GetLayerData();
	const BatchNormalization_LayerData_Base& GetLayerData()const;


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 WriteToBuffer(BYTE* o_lpBuffer)const;


public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn(unsigned int batchSize);

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate(unsigned int batchSize);


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculateLoop();


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer);

	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

public:
	//================================
	// �w�K����
	//================================
	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K���������C���[�ɔ��f������.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		�o�͌덷�����A���͌덷�����͒��O��CalculateLearnError�̒l���Q�Ƃ���. */
	ErrorCode ReflectionLearnError(void);

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const;
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif