//======================================
// �����֐����C���[
// GPU�����p
//======================================
#ifndef __ACTIVATION_CPU_H__
#define __ACTIVATION_CPU_H__

#include"stdafx.h"

#include"Activation_DATA.hpp"
#include"Activation_FUNC.hpp"
#include"Activation_Base.h"

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

class Activation_GPU : public Activation_Base
{
private:
	// �f�[�^�{��
	class Activation_LayerData_GPU& layerData;

	// ���o�̓o�b�t�@
	thrust::device_vector<F32>				lpOutputBuffer_d;		/**< �o�̓o�b�t�@ <�o�b�`��><���͐M����> */
	thrust::device_vector<F32>				lpDInputBuffer_d;		/**< ���͌덷���� <�o�b�`��><���͐M����> */

	/**< �o�b�`�����p���͌덷���� <�o�b�`��> */

	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	// ���Z���̓��̓f�[�^
	CONST_BATCH_BUFFER_POINTER m_lpInputBuffer_d;
	CONST_BATCH_BUFFER_POINTER m_lpDOutputBufferPrev_d;


	// ���Z�����p�̃o�b�t�@
	cudnnHandle_t cudnnHandle;
    cudnnActivationDescriptor_t  activDesc;
	cudnnTensorDescriptor_t inputTensorDesc;
	cudnnTensorDescriptor_t outputTensorDesc;

public:
	/** �R���X�g���N�^ */
	Activation_GPU(Gravisbell::GUID guid, class Activation_LayerData_GPU& i_layerData);
	/** �f�X�g���N�^ */
	virtual ~Activation_GPU();

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
	Activation_LayerData_Base& GetLayerData();
	const Activation_LayerData_Base& GetLayerData()const;


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
	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

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