//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌����������C���[
// CPU�����p
//======================================
#ifndef __CONVOLUTION_CPU_H__
#define __CONVOLUTION_CPU_H__

#include"stdafx.h"

#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Convolution_CPU : public Convolution_Base
{
private:
	// �f�[�^�{��
	class Convolution_LayerData_CPU& layerData;

	// ���o�̓o�b�t�@
	std::vector<F32>			lpOutputBuffer;		/**< �o�̓o�b�t�@ <�o�b�`��><��ݍ��ݐ�> */

	std::vector<F32*>						lppBatchOutputBuffer;		/**< �o�b�`�����p�o�̓o�b�t�@ <�o�b�`��> */

	std::vector<F32>			lpDNeuron;	/**< �j���[�����̊w�K�� */
	std::vector<F32>			lpDBias;	/**< �o�C�A�X�̊w�K�� */

	// Get�֐����g���Ə����s�������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 filterSize;						/**< �t�B���^�T�C�Y */
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 neuronCount;					/**< �j���[������ */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	IODataStruct paddingInputDataStruct;	/**< �p�f�B���O��̓��̓o�b�t�@�̓��̓f�[�^�\�� */
	std::vector<std::vector<F32>> lpPaddingInputBuffer;	/**< �p�f�B���O��̓��̓o�b�t�@ <�o�b�`��><���̓o�b�t�@> */

	// ���Z���̓��̓f�[�^
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppInputBuffer;		/**< ���Z���̓��̓f�[�^ */
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppDOutputBuffer;		/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */

	BATCH_BUFFER_POINTER m_lpDInputBuffer;
	std::vector<BATCH_BUFFER_POINTER> m_lppDInputBuffer;			/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */

	// ���Z�����p�̃o�b�t�@


public:
	/** �R���X�g���N�^ */
	Convolution_CPU(Gravisbell::GUID guid, class Convolution_LayerData_CPU& i_layerData);
	/** �f�X�g���N�^ */
	virtual ~Convolution_CPU();


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
	Convolution_LayerData_Base& GetLayerData();
	const Convolution_LayerData_Base& GetLayerData()const;


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
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

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