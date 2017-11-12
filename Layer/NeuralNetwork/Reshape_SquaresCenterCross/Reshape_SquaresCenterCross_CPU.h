//======================================
// �o�͐M���������C���[
// CPU�����p
//======================================
#ifndef __RESHAPE_SQUARECENTERCROSS_CPU_H__
#define __RESHAPE_SQUARECENTERCROSS_CPU_H__

#include"stdafx.h"

#include"Reshape_SquaresCenterCross_DATA.hpp"
#include"Reshape_SquaresCenterCross_FUNC.hpp"
#include"Reshape_SquaresCenterCross_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Reshape_SquaresCenterCross_CPU : public Reshape_SquaresCenterCross_Base
{
private:
	// �f�[�^�{��
	class Reshape_SquaresCenterCross_LayerData_CPU& layerData;


	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	std::vector<U32>						m_lpConvertTable;		/**< �ϊ��e�[�u��(sqrt(x-1)*2+1)^2 */
	std::vector<U32*>						m_lppConvertTable;		/**< �ϊ��e�[�u��<sqrt(x-1)*2+1><sqrt(x-1)*2+1> */

	// ���o�̓o�b�t�@
	CONST_BATCH_BUFFER_POINTER				m_lpInputBuffer;		/**< ���̓o�b�t�@ <�o�b�`�� * ���͐M����> */
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppInputBuffer;		/**< ���̓o�b�t�@ <�o�b�`��><���͐M����>  */

	std::vector<F32>						m_lpOutputBuffer;		/**< �o�̓o�b�t�@ <�o�b�`�� * �o�͐M����> */
	std::vector<BATCH_BUFFER_POINTER>		m_lppOutputBuffer;		/**< �o�̓o�b�t�@ <�o�b�`��><�o�͐M����> */

	CONST_BATCH_BUFFER_POINTER				m_lpDOutputBuffer;		/**< �o�͌덷�o�b�t�@ <�o�b�`�� * �o�͐M����> */
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppDOutputBuffer;		/**< �o�͌덷�o�b�t�@ <�o�b�`��><�o�͐M����> */

	BATCH_BUFFER_POINTER					m_lpDInputBuffer;		/**< ���͌덷�o�b�t�@ <�o�b�`�� * ���͐M����> */
	std::vector<BATCH_BUFFER_POINTER>		m_lppDInputBuffer;		/**< ���͌덷�o�b�t�@ <�o�b�`��><���͐M����> */


public:
	/** �R���X�g���N�^ */
	Reshape_SquaresCenterCross_CPU(Gravisbell::GUID guid, class Reshape_SquaresCenterCross_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct);
	/** �f�X�g���N�^ */
	virtual ~Reshape_SquaresCenterCross_CPU();

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
	Reshape_SquaresCenterCross_LayerData_Base& GetLayerData();
	const Reshape_SquaresCenterCross_LayerData_Base& GetLayerData()const;


public:
	//================================
	// ���Z����
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
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz���[GetOutputDataCount()]�z��
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